# Hardware H.264 decode on the GPU via VK_KHR_video_decode_queue.
# Input: an H.264 Annex-B elementary stream (SPS/PPS/slice NAL units).
# Output: luma (Y) planes in display order. Everything (session, DPB reference
# management, POC ordering) is driven through VulkanCore's raw video-decode FFI —
# Vulkan.jl's high-level wrapper does not generate the *video codec* API. A
# minimal exp-Golomb parser reads the SPS/PPS and per-slice headers to build the
# StdVideoH264* structs and manage the decoded-picture buffer.
#
# The decoded *frames*, though, are ordinary Vulkan images: the decoder produces
# a [`VideoImage`] per frame (a real `Vulkan.Image`), and moving a frame's luma
# onto a `LavaArray` is a normal `copyto!(dst, img)` through the high-level
# `cmd_copy_image_to_buffer` — no raw FFI on the resource/transfer side.

# ============================================================================
# VideoImage — a decoded video frame resident in a GPU image
# ============================================================================
"""
    VideoImage

One hardware-decoded video frame, resident in a device-local multi-planar
(NV12) `Vulkan.Image`. It is a first-class Lava image handle:

  * `copyto!(dst::LavaArray{UInt8}, img)` copies the luma (Y) plane into a
    device-local array (image→buffer, entirely on the GPU — no host round-trip).
  * `Array(img)` downloads the luma plane to a host `Matrix{UInt8}`.
  * `size(img)` is the display (cropped) size; `eltype(img) == UInt8`.

Produced by the hardware H.264 decoder — see [`decode_h264_gpu`](@ref).
"""
mutable struct VideoImage
    image::Vulkan.Image
    memory::Vulkan.DeviceMemory
    codedwidth::Int
    codedheight::Int
    width::Int      # display (cropped) width
    height::Int     # display (cropped) height
    format::Vulkan.Format
    # Current VkImageLayout as a raw UInt32 so the decode loop's in-place
    # barriers and the high-level copy path share one source of truth.
    layout::Base.RefValue{UInt32}
end
Base.size(img::VideoImage) = (img.width, img.height)
Base.eltype(::VideoImage) = UInt8
Base.show(io::IO, img::VideoImage) = print(io, "VideoImage($(img.width)×$(img.height), NV12)")

# Record a luma (Y-plane) image→buffer copy of `img` (display size, top-left,
# tightly packed) into `buffer` at byte `offset`, on high-level command buffer
# `cb`. `img` must already be TRANSFER_SRC_OPTIMAL.
function record_luma_copy!(cb, img::VideoImage, buffer::Vulkan.Buffer, offset::Integer)
    Vulkan.cmd_copy_image_to_buffer(cb, img.image,
        Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
        [Vulkan.BufferImageCopy(UInt64(offset), 0, 0,
            Vulkan.ImageSubresourceLayers(Vulkan.IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1),
            Vulkan.Offset3D(0, 0, 0), Vulkan.Extent3D(img.width, img.height, 1))])
    return cb
end

# Record the chroma (interleaved U/V) plane of an NV12 frame into `buffer`: a
# (width÷2 × height÷2) region of 2-byte (U,V) pixels, tightly packed, so the
# destination is a `width × (height÷2)` UInt8 buffer of interleaved U,V samples.
function record_chroma_copy!(cb, img::VideoImage, buffer::Vulkan.Buffer, offset::Integer)
    Vulkan.cmd_copy_image_to_buffer(cb, img.image,
        Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
        [Vulkan.BufferImageCopy(UInt64(offset), 0, 0,
            Vulkan.ImageSubresourceLayers(Vulkan.IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1),
            Vulkan.Offset3D(0, 0, 0), Vulkan.Extent3D(img.width ÷ 2, img.height ÷ 2, 1))])
    return cb
end

# Transition `img` to TRANSFER_SRC_OPTIMAL on `cb` if it isn't already there.
function transition_transfer_src!(cb, img::VideoImage)
    img.layout[] == UInt32(Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) && return cb
    Vulkan.cmd_pipeline_barrier(cb, [], [],
        [Vulkan.ImageMemoryBarrier(Vulkan.ACCESS_MEMORY_WRITE_BIT,
            Vulkan.ACCESS_TRANSFER_READ_BIT, Vulkan.ImageLayout(img.layout[]),
            Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            Vulkan.QUEUE_FAMILY_IGNORED, Vulkan.QUEUE_FAMILY_IGNORED, img.image,
            Vulkan.ImageSubresourceRange(Vulkan.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))];
        src_stage_mask = Vulkan.PIPELINE_STAGE_ALL_COMMANDS_BIT,
        dst_stage_mask = Vulkan.PIPELINE_STAGE_TRANSFER_BIT)
    img.layout[] = UInt32(Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    return cb
end

"""
    copyto!(dst::LavaArray{UInt8}, img::VideoImage) -> dst

Copy the decoded frame's luma (Y) plane into the device-local array `dst`
(image→buffer, entirely on the GPU). `dst` must hold at least `prod(size(img))`
bytes and be allocated with a TRANSFER_DST usage flag. Runs one command buffer
on the video-decode queue and blocks until it lands.
"""
function Base.copyto!(dst::LavaArray{UInt8}, img::VideoImage)
    need = img.width * img.height
    length(dst) >= need || error("VideoImage copyto!: dst holds $(length(dst)) < $need bytes")
    mbuf = dst.buf[]
    ctx = mbuf.ctx::VkContext
    dev = ctx.device
    pool = @vk_checked "video_copy_pool" Vulkan.create_command_pool(dev, ctx.video_decode_queue_family_index)
    cb = first(@vk_checked "video_copy_cb" Vulkan.allocate_command_buffers(dev,
        Vulkan.CommandBufferAllocateInfo(pool, Vulkan.COMMAND_BUFFER_LEVEL_PRIMARY, 1)))
    @vk_checked "video_copy_begin" Vulkan.begin_command_buffer(cb, Vulkan.CommandBufferBeginInfo())
    transition_transfer_src!(cb, img)
    record_luma_copy!(cb, img, mbuf.buffer, mbuf.pool_offset + dst.offset)
    @vk_checked "video_copy_end" Vulkan.end_command_buffer(cb)
    @vk_checked "video_copy_submit" Vulkan.queue_submit(ctx.video_decode_queue, [Vulkan.SubmitInfo([], [], [cb], [])])
    @vk_checked "video_copy_wait" Vulkan.queue_wait_idle(ctx.video_decode_queue)
    return dst
end

"""
    Array(img::VideoImage) -> Matrix{UInt8}

Download the decoded frame's luma (Y) plane to the host as a `width × height`
`Matrix{UInt8}` (grayscale = the NV12 Y plane).
"""
function Base.Array(img::VideoImage)
    dst = LavaArray{UInt8,2}(undef, (img.width, img.height);
                             extra_usage = UInt32(Vulkan.BUFFER_USAGE_TRANSFER_DST_BIT))
    copyto!(dst, img)
    return Array(dst)
end

module VideoDecode
import Vulkan
const Vk = Vulkan
const C = Vk.VkCore

# ---------- bitstream ----------
mutable struct BR; d::Vector{UInt8}; p::Int; end
rb1(b)=(byte=b.d[(b.p>>3)+1]; v=(byte>>(7-(b.p&7)))&0x01; b.p+=1; Int(v))
rbn(b,n)=(v=0; for _ in 1:n; v=(v<<1)|rb1(b); end; v)
ue(b)=(z=0; while b.p<8*length(b.d) && rb1(b)==0; z+=1; end; z==0 ? 0 : ((1<<z)-1+rbn(b,z)))
se(b)=(k=ue(b); iseven(k) ? -(k>>1) : (k+1)>>1)
function unescape(d)
    out=UInt8[]; z=0
    for b in d
        if z>=2 && b==0x03; z=0; continue; end
        push!(out,b); z = b==0 ? z+1 : 0
    end; out
end
function split_nals(d)
    out=Tuple{Int,Int,Vector{UInt8}}[]; starts=Int[]; i=1
    while i<=length(d)-3
        if d[i]==0 && d[i+1]==0 && d[i+2]==1; push!(starts,i+3); i+=3 else i+=1 end
    end
    for (k,st) in enumerate(starts)
        e = k<length(starts) ? starts[k+1]-4 : length(d)
        while e>st && d[e]==0; e-=1 end
        nal=d[st:e]; push!(out,(Int(nal[1]&0x1f), Int((nal[1]>>5)&3), nal))
    end
    out
end
pbytes(v::UInt32)=(UInt8(v&0xff),UInt8((v>>8)&0xff),UInt8((v>>16)&0xff),UInt8((v>>24)&0xff))
packflags(vals)=(v=UInt32(0); for (i,x) in enumerate(vals); v|=(UInt32(x&1)<<(i-1)); end; v)
const HIGHSET=Set([100,110,122,244,44,83,86,118,128,138,139,134,135])

function parse_sps(nal)
    b=BR(unescape(nal[2:end]),0)
    profile=rbn(b,8); cflags=rbn(b,8); level=rbn(b,8); sps_id=ue(b)
    chroma=1; sepcol=0; bdl=0; bdc=0; qpp=0; scal=0
    if profile in HIGHSET
        chroma=ue(b); chroma==3 && (sepcol=rb1(b)); bdl=ue(b); bdc=ue(b); qpp=rb1(b); scal=rb1(b)
        scal==1 && error("seq scaling matrix not handled")
    end
    log2fn=ue(b); poct=ue(b); log2poc=0; dpoaz=0
    if poct==0; log2poc=ue(b)
    elseif poct==1; dpoaz=rb1(b); se(b); se(b); nc=ue(b); for _ in 1:nc; se(b); end; end
    maxref=ue(b); gaps=rb1(b); wmbs=ue(b); hmap=ue(b); fmo=rb1(b)
    mbaff = fmo==0 ? rb1(b) : 0; direct8=rb1(b); crop=rb1(b); cl=cr=ct=cb=0
    crop==1 && (cl=ue(b);cr=ue(b);ct=ue(b);cb=ue(b)); vui=rb1(b)
    (;profile,cflags,level,sps_id,chroma,sepcol,bdl,bdc,qpp,log2fn,poct,log2poc,dpoaz,maxref,gaps,wmbs,hmap,fmo,mbaff,direct8,crop,cl,cr,ct,cb,vui)
end
function parse_pps(nal)
    b=BR(unescape(nal[2:end]),0)
    pps_id=ue(b); sps_id=ue(b); entropy=rb1(b); bottom=rb1(b); nsg=ue(b)
    nsg>0 && error("slice groups not handled")
    nl0=ue(b); nl1=ue(b); wp=rb1(b); wbi=rbn(b,2); qp=se(b); qs=se(b); cqp=se(b)
    dbf=rb1(b); cintra=rb1(b); redun=rb1(b); t8=0; cqp2=cqp
    if b.p < 8*length(b.d)-8; t8=rb1(b); ps=rb1(b); ps==1 && error("pic scaling not handled"); cqp2=se(b); end
    (;pps_id,sps_id,entropy,bottom,nl0,nl1,wp,wbi,qp,qs,cqp,dbf,cintra,redun,t8,cqp2)
end
# slice header (enough for POC/DPB): frame_mbs_only assumed
function parse_slice(nal, sps)
    nrid=Int((nal[1]>>5)&3); idr=(nal[1]&0x1f)==5
    b=BR(unescape(nal[2:end]),0)
    first_mb=ue(b); stype=ue(b)%5; ppsid=ue(b)
    sps.sepcol==1 && rbn(b,2)
    fnum=rbn(b, sps.log2fn+4)
    idrid = idr ? ue(b) : 0
    poclsb = sps.poct==0 ? rbn(b, sps.log2poc+4) : 0
    (;nrid,idr,first_mb,stype,fnum,idrid,poclsb)
end

# dec_ref_pic_marking (7.3.3.3): the memory-management control operations a
# reference picture carries. Returns (adaptive::Bool, ops::Vector{(op,arg,ltidx)}).
# Reaching this syntax means fully parsing the slice header up to it — num_ref_idx
# overrides, ref_pic_list_modification, and (skipping) pred_weight_table. Without
# this, hierarchical-B (B-pyramid) streams desync the DPB: the encoder uses MMCO
# op 1 to drop transient reference B-frames while keeping anchors, which a plain
# sliding window gets wrong. Only meaningful for reference slices (nal_ref_idc≠0).
function parse_mmco(nal, sps, pps)
    nrid=Int((nal[1]>>5)&3)
    nrid==0 && return (false, Tuple{Int,Int,Int}[])
    idr=(nal[1]&0x1f)==5
    b=BR(unescape(nal[2:end]),0)
    ue(b); stype=ue(b)%5; ue(b)                       # first_mb, slice_type, pps_id
    sps.sepcol==1 && rbn(b,2)
    rbn(b, sps.log2fn+4)                              # frame_num
    idr && ue(b)                                      # idr_pic_id
    if sps.poct==0
        rbn(b, sps.log2poc+4); pps.bottom==1 && se(b)
    elseif sps.poct==1 && sps.dpoaz==0
        se(b); pps.bottom==1 && se(b)
    end
    pps.redun==1 && ue(b)                             # redundant_pic_cnt
    stype==1 && rb1(b)                                # direct_spatial_mv_pred_flag
    nrl0=pps.nl0+1; nrl1=pps.nl1+1
    if stype==0 || stype==1                           # num_ref_idx_active_override
        if rb1(b)==1; nrl0=ue(b)+1; stype==1 && (nrl1=ue(b)+1); end
    end
    modlist()=(rb1(b)==1 && while true; idc=ue(b); idc==3 && break; (idc==0||idc==1||idc==2) && ue(b); end)
    stype!=2 && stype!=4 && modlist()                 # ref_pic_list_modification l0
    stype==1 && modlist()                             # ref_pic_list_modification l1
    if (pps.wp==1 && stype==0) || (pps.wbi==1 && stype==1)   # pred_weight_table (skip)
        ue(b); sps.chroma!=0 && ue(b)
        for cnt in (stype==1 ? (nrl0,nrl1) : (nrl0,))
            for _ in 1:cnt
                rb1(b)==1 && (se(b); se(b))
                sps.chroma!=0 && rb1(b)==1 && (se(b);se(b);se(b);se(b))
            end
        end
    end
    if idr; rb1(b); rb1(b); return (false, Tuple{Int,Int,Int}[]); end
    adaptive=rb1(b)==1
    ops=Tuple{Int,Int,Int}[]
    if adaptive
        while true
            op=ue(b); op==0 && break
            arg=0; lt=0
            (op==1||op==3) && (arg=ue(b))             # difference_of_pic_nums_minus1
            op==2 && (arg=ue(b))                      # long_term_pic_num
            (op==3||op==6) && (lt=ue(b))              # long_term_frame_idx
            op==4 && (arg=ue(b))                      # max_long_term_frame_idx_plus1
            push!(ops,(op,arg,lt))
            length(ops)>64 && break
        end
    end
    (adaptive, ops)
end

const LEVELMAP=Dict(10=>0,11=>1,12=>2,13=>3,20=>4,21=>5,22=>6,30=>7,31=>8,32=>9,40=>10,41=>11,42=>12,50=>13,51=>14,52=>15,60=>16,61=>17,62=>18)
function std_sps(s)
    csf(i)=(s.cflags>>(7-i))&1
    fl=packflags((csf(0),csf(1),csf(2),csf(3),csf(4),csf(5),s.direct8,s.mbaff,s.fmo,s.dpoaz,s.sepcol,s.gaps,s.qpp,s.crop,0,s.vui))
    C.StdVideoH264SequenceParameterSet(C.StdVideoH264SpsFlags(pbytes(fl)),
        C.StdVideoH264ProfileIdc(s.profile), C.StdVideoH264LevelIdc(LEVELMAP[s.level]),
        C.StdVideoH264ChromaFormatIdc(s.chroma), UInt8(s.sps_id), UInt8(s.bdl), UInt8(s.bdc),
        UInt8(s.log2fn), C.StdVideoH264PocType(s.poct), Int32(0),Int32(0), UInt8(s.log2poc), UInt8(0),
        UInt8(s.maxref), UInt8(0), UInt32(s.wmbs), UInt32(s.hmap), UInt32(s.cl),UInt32(s.cr),UInt32(s.ct),UInt32(s.cb),
        UInt32(0), Ptr{Int32}(C_NULL), Ptr{C.StdVideoH264ScalingLists}(C_NULL), Ptr{C.StdVideoH264SequenceParameterSetVui}(C_NULL))
end
function std_pps(p)
    fl=packflags((p.t8,p.redun,p.cintra,p.dbf,p.wp,p.bottom,p.entropy,0))
    C.StdVideoH264PictureParameterSet(C.StdVideoH264PpsFlags(pbytes(fl)), UInt8(p.sps_id),UInt8(p.pps_id),
        UInt8(p.nl0),UInt8(p.nl1), C.StdVideoH264WeightedBipredIdc(p.wbi),
        Int8(p.qp),Int8(p.qs),Int8(p.cqp),Int8(p.cqp2), Ptr{C.StdVideoH264ScalingLists}(C_NULL))
end

# ---------- Vulkan helpers ----------
struct Ctxwrap; ctx; dev; qf::UInt32; q; end
dfp(w,n)=Vk.function_pointer(w.dev,n)
rp(x)=Base.unsafe_convert(Ptr{eltype(x)},x); pc(x)=Ptr{Cvoid}(rp(x))
function memtype(w,bits,want)
    mp=w.ctx.memory_properties
    for i in 0:(Int(mp.memory_type_count)-1)
        (bits&(UInt32(1)<<i))!=0 && (UInt32(mp.memory_types[i+1].property_flags)&UInt32(want))==UInt32(want) && return i
    end
    for i in 0:(Int(mp.memory_type_count)-1); (bits&(UInt32(1)<<i))!=0 && return i; end
    error("no memory type")
end

# returns (video-decode profile ref, profile-list ref, caps NamedTuple)
function video_profile(w)
    hp=Ref(C.VkVideoDecodeH264ProfileInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,C_NULL,C.STD_VIDEO_H264_PROFILE_IDC_HIGH,C.VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR))
    pr=Ref(C.VkVideoProfileInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR, Ptr{Cvoid}(rp(hp)), C.VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR, UInt32(C.VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR), UInt32(C.VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR), UInt32(C.VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)))
    pl=Ref(C.VkVideoProfileListInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR, C_NULL, UInt32(1), rp(pr)))
    (hp,pr,pl)
end

"""
    decode_h264(ctx, annexb; maxframes=typemax(Int)) -> (w, h, Vector{VideoImage-backed LavaArray})

Hardware-decode an H.264 Annex-B stream on the GPU. Each decoded frame's luma (Y)
plane is left GPU-resident in its own device-local `LavaArray{UInt8,2}` (cropped to
the display size, copied out through the high-level [`VideoImage`] `copyto!` path —
no host round-trip). Frames are returned in DISPLAY order. Grayscale = the NV12 Y
plane. The video *codec* commands (session/decode/DPB) are raw VulkanCore FFI —
Vulkan.jl generates no wrappers for the video codec API — but the frame images and
the transfer are ordinary Vulkan.jl abstractions.
"""
function decode_h264(ctx, annexb::Vector{UInt8}; maxframes::Int=typemax(Int), chroma::Bool=false)
    ctx.video_decode_available || error("device has no video decode support")
    Lava = parentmodule(@__MODULE__)   # parent: LavaArray, VideoImage, record_luma_copy!
    w=Ctxwrap(ctx, ctx.device, UInt32(ctx.video_decode_queue_family_index), ctx.video_decode_queue)
    dev=w.dev
    nalu=split_nals(annexb)
    sps = parse_sps(first(n[3] for n in nalu if n[1]==7))
    pps = parse_pps(first(n[3] for n in nalu if n[1]==8))
    # Only 4:2:0 (NV12) is supported: it's the one chroma format the H.264 video
    # decode engine handles, and the output/DPB images are hardcoded NV12. A
    # 4:2:2/4:4:4 stream would silently mis-decode, so reject it explicitly.
    sps.chroma == 1 ||
        error("decode_h264: only 4:2:0 chroma is supported (stream is chroma_format_idc=$(sps.chroma); " *
              "profile_idc=$(sps.profile)). Transcode to yuv420p first.")
    # Streams with max_num_ref_frames ≤ 1 (baseline single-reference / all-intra)
    # currently decode to garbage on this path; the common camera/delivery case
    # (multi-reference High profile) is pixel-exact. Reject rather than mis-decode.
    sps.maxref >= 2 ||
        error("decode_h264: streams with max_num_ref_frames=$(sps.maxref) (single-reference/all-intra) " *
              "are not yet supported; re-encode with -refs 2 or higher.")
    CW=(sps.wmbs+1)*16; CH=(sps.hmap+1)*16
    DW=CW-2*sps.cr; DH=CH-2*sps.cb   # display size (chroma 420 crop units = 2px)
    fmt=C.VkFormat(1000156003)       # G8_B8R8_2PLANE_420_UNORM (NV12)
    PIN=Any[]; pin(x)=(push!(PIN,x);x)
    hp,pr,pl=video_profile(w); pin(hp);pin(pr);pin(pl); pProf=rp(pr)
    # session
    rHdr=pin(Ref(C.VkExtensionProperties(ntuple(_->Cchar(0),256),UInt32(0))))
    # get stdHeaderVersion from caps
    let hc=Ref(C.VkVideoDecodeH264CapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR,C_NULL,C.StdVideoH264LevelIdc(0),C.VkOffset2D(0,0))),
        dc=Ref(C.VkVideoDecodeCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,Ptr{Cvoid}(rp(hc)),UInt32(0))),
        e0=C.VkExtent2D(0,0), cp=Ref(C.VkVideoCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,Ptr{Cvoid}(rp(dc)),UInt32(0),UInt64(0),UInt64(0),e0,e0,e0,UInt32(0),UInt32(0),C.VkExtensionProperties(ntuple(_->Cchar(0),256),UInt32(0))))
        GC.@preserve hc dc cp PIN ccall(Vk.function_pointer(ctx.instance,"vkGetPhysicalDeviceVideoCapabilitiesKHR"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),ctx.physical_device.vks,Ptr{Cvoid}(pProf),pc(cp))
        rHdr[]=cp[].stdHeaderVersion
    end
    maxslots=UInt32(sps.maxref+2)
    sci=pin(Ref(C.VkVideoSessionCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,C_NULL,w.qf,UInt32(0),pProf,fmt,C.VkExtent2D(CW,CH),fmt,maxslots,UInt32(sps.maxref),rp(rHdr))))
    rSess=Ref{C.VkVideoSessionKHR}(); GC.@preserve PIN ccall(dfp(w,"vkCreateVideoSessionKHR"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(sci),C_NULL,rp(rSess)); SESSION=rSess[]
    # bind session memory
    let mc=Ref(UInt32(0)); ccall(dfp(w,"vkGetVideoSessionMemoryRequirementsKHR"),Int32,(Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{UInt32},Ptr{Cvoid}),dev.vks,SESSION,mc,C_NULL)
        mreqs=[C.VkVideoSessionMemoryRequirementsKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR,C_NULL,UInt32(0),C.VkMemoryRequirements(UInt64(0),UInt64(0),UInt32(0))) for _ in 1:Int(mc[])]
        GC.@preserve mreqs ccall(dfp(w,"vkGetVideoSessionMemoryRequirementsKHR"),Int32,(Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{UInt32},Ptr{Cvoid}),dev.vks,SESSION,mc,pointer(mreqs))
        binds=C.VkBindVideoSessionMemoryInfoKHR[]
        for mr in mreqs
            m=Vk.unwrap(Vk.allocate_memory(dev,mr.memoryRequirements.size,memtype(w,mr.memoryRequirements.memoryTypeBits,C.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))); pin(m)
            push!(binds,C.VkBindVideoSessionMemoryInfoKHR(C.VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,C_NULL,mr.memoryBindIndex,m.vks,UInt64(0),mr.memoryRequirements.size))
        end
        GC.@preserve binds PIN ccall(dfp(w,"vkBindVideoSessionMemoryKHR"),Int32,(Ptr{Cvoid},C.VkVideoSessionKHR,UInt32,Ptr{Cvoid}),dev.vks,SESSION,UInt32(length(binds)),pointer(binds))
    end
    # session params (SPS+PPS)
    rSPSs=pin(Ref(std_sps(sps))); rPPSs=pin(Ref(std_pps(pps)))
    add=pin(Ref(C.VkVideoDecodeH264SessionParametersAddInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,C_NULL,UInt32(1),rp(rSPSs),UInt32(1),rp(rPPSs))))
    h264c=pin(Ref(C.VkVideoDecodeH264SessionParametersCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,C_NULL,UInt32(1),UInt32(1),rp(add))))
    pci=pin(Ref(C.VkVideoSessionParametersCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,Ptr{Cvoid}(rp(h264c)),UInt32(0),C_NULL,SESSION)))
    rParams=Ref{C.VkVideoSessionParametersKHR}(); GC.@preserve PIN ccall(dfp(w,"vkCreateVideoSessionParametersKHR"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(pci),C_NULL,rp(rParams)); PARAMS=rParams[]
    # DPB image pool — real Vulkan.Image handles wrapped as VideoImages. The
    # decode commands (raw FFI) take the raw `.vks` handles; the output copy uses
    # the high-level VideoImage path. High-level handles are refcounted, so the
    # whole pool auto-frees when this call returns.
    nslots=Int(maxslots)
    fmt_hl=Vk.Format(1000156003)   # G8_B8R8_2PLANE_420_UNORM (NV12)
    # DPB images MUST advertise the SAME video profile the session was created
    # with, or the driver silently declines to decode into them. Reuse the exact
    # raw `pl` (VkVideoProfileListInfoKHR) pointer the session uses — high-level
    # `create_image` accepts a raw pNext pointer, so no re-serialization can drift.
    dpbusage=Vk.IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR|Vk.IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR|Vk.IMAGE_USAGE_TRANSFER_SRC_BIT
    imgs=Vector{Any}(undef,nslots)  # (vimg::VideoImage, view::Vulkan.ImageView)
    for k in 1:nslots
        image=GC.@preserve PIN Vk.Image(dev, Vk.IMAGE_TYPE_2D, fmt_hl, Vk.Extent3D(CW,CH,1),
            1, 1, Vk.SAMPLE_COUNT_1_BIT, Vk.IMAGE_TILING_OPTIMAL, dpbusage,
            Vk.SHARING_MODE_EXCLUSIVE, UInt32[], Vk.IMAGE_LAYOUT_UNDEFINED; next=Ptr{Cvoid}(rp(pl)))
        mem=Lava.alloc_image_memory(ctx, image)
        view=Vk.ImageView(dev, image, Vk.IMAGE_VIEW_TYPE_2D, fmt_hl,
            Vk.ComponentMapping(Vk.COMPONENT_SWIZZLE_IDENTITY,Vk.COMPONENT_SWIZZLE_IDENTITY,Vk.COMPONENT_SWIZZLE_IDENTITY,Vk.COMPONENT_SWIZZLE_IDENTITY),
            Vk.ImageSubresourceRange(Vk.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
        vimg=Lava.VideoImage(image, mem, CW, CH, DW, DH, fmt_hl, Ref(UInt32(C.VK_IMAGE_LAYOUT_UNDEFINED)))
        pin(image); pin(mem); pin(view)
        imgs[k]=(vimg, view)
    end
    # bitstream buffer (reused, sized to the largest AU) + readback buffer (Y+UV)
    maxau=0; for n in nalu; (n[1]==1||n[1]==5) && (maxau=max(maxau,length(n[3])+3)); end
    bufsz=UInt64(cld(maxau,256)*256)
    bbuf,bmem,bmap = let
        ci=Ref(C.VkBufferCreateInfo(C.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,Ptr{Cvoid}(rp(pl)),UInt32(0),bufsz,UInt32(C.VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR),C.VK_SHARING_MODE_EXCLUSIVE,UInt32(0),Ptr{UInt32}(C_NULL)))
        rb=Ref{C.VkBuffer}(); GC.@preserve ci pl PIN ccall(dfp(w,"vkCreateBuffer"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(ci),C_NULL,rp(rb)); bf=rb[]
        mr=Ref(C.VkMemoryRequirements(UInt64(0),UInt64(0),UInt32(0))); ccall(dfp(w,"vkGetBufferMemoryRequirements"),Cvoid,(Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}),dev.vks,bf,rp(mr))
        m=Vk.unwrap(Vk.allocate_memory(dev,mr[].size,memtype(w,mr[].memoryTypeBits,UInt32(C.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)|UInt32(C.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)))); pin(m)
        ccall(dfp(w,"vkBindBufferMemory"),Int32,(Ptr{Cvoid},C.VkBuffer,C.VkDeviceMemory,UInt64),dev.vks,bf,m.vks,UInt64(0))
        pmap=Ref{Ptr{Cvoid}}(); ccall(dfp(w,"vkMapMemory"),Int32,(Ptr{Cvoid},C.VkDeviceMemory,UInt64,UInt64,UInt32,Ptr{Cvoid}),dev.vks,m.vks,UInt64(0),bufsz,UInt32(0),Base.unsafe_convert(Ptr{Ptr{Cvoid}},pmap))
        (bf,m,Ptr{UInt8}(pmap[]))
    end
    # command pool + buffer. The command buffer is a *high-level* Vulkan handle
    # so the output copy can use `cmd_copy_image_to_buffer`; the raw video-codec
    # commands take its `.vks` raw handle.
    cbh=first(Vk.unwrap(Vk.allocate_command_buffers(dev,
        Vk.CommandBufferAllocateInfo(Vk.unwrap(Vk.create_command_pool(dev, w.qf; flags=Vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)),
            Vk.COMMAND_BUFFER_LEVEL_PRIMARY, 1)))); pin(cbh)
    CB=cbh.vks

    # ---- group slice NALs into access units (new AU when first_mb==0) ----
    aus=Vector{Vector{Vector{UInt8}}}()
    for n in nalu
        (n[1]==1||n[1]==5) || continue
        sh=parse_slice(n[3],sps)
        if sh.first_mb==0; push!(aus,[n[3]]) else push!(aus[end],n[3]) end
    end

    # ---- decode loop ----
    allcol=C.VkImageSubresourceRange(UInt32(C.VK_IMAGE_ASPECT_COLOR_BIT),UInt32(0),UInt32(1),UInt32(0),UInt32(1))
    IGN=UInt32(0xffffffff); allst=UInt32(C.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
    barrier(img,old,new)=Ref(C.VkImageMemoryBarrier(C.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,C_NULL,UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT),UInt32(C.VK_ACCESS_MEMORY_READ_BIT)|UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT),C.VkImageLayout(old),C.VkImageLayout(new),IGN,IGN,img,allcol))
    emit(cb,b)=ccall(dfp(w,"vkCmdPipelineBarrier"),Cvoid,(C.VkCommandBuffer,UInt32,UInt32,UInt32,UInt32,Ptr{Cvoid},UInt32,Ptr{Cvoid},UInt32,Ptr{Cvoid}),cb,allst,allst,UInt32(0),UInt32(0),C_NULL,UInt32(0),C_NULL,UInt32(1),pc(b))
    picres(view)=C.VkVideoPictureResourceInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,C_NULL,C.VkOffset2D(0,0),C.VkExtent2D(CW,CH),UInt32(0),view)
    DPBLAYOUT=UInt32(C.VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR); TSRC=UInt32(C.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    dpb=Tuple{Int,Int,Int}[]   # (slot, framenum, poc)  ; slot indexes imgs
    freeslots=collect(1:nslots)
    prevmsb=0; prevlsb=0; maxpoclsb=1<<(sps.log2poc+4); gop=0
    outputs=Tuple{Int,Int,Any,Any}[]   # (gop, poc, Y::LavaArray, UV::LavaArray|nothing)
    isfirst=true
    for (ai_i,au) in enumerate(aus)
        ai_i>maxframes && break
        sh=parse_slice(au[1],sps)
        if sh.idr
            for (slot,_,_) in dpb; push!(freeslots,slot); end; empty!(dpb); prevmsb=0; prevlsb=0; gop+=1
        end
        # POC (type 0)
        if sps.poct==0
            if sh.idr; pmsb=0
            elseif sh.poclsb<prevlsb && (prevlsb-sh.poclsb)>=maxpoclsb÷2; pmsb=prevmsb+maxpoclsb
            elseif sh.poclsb>prevlsb && (sh.poclsb-prevlsb)>maxpoclsb÷2; pmsb=prevmsb-maxpoclsb
            else; pmsb=prevmsb; end
            poc=pmsb+sh.poclsb
            if sh.nrid!=0; prevmsb=pmsb; prevlsb=sh.poclsb; end
        else; poc=ai_i-1; end
        isref = sh.nrid!=0
        outslot=popfirst!(freeslots)
        outvimg,outviewhl=imgs[outslot]
        outimg=outvimg.image.vks; outview=outviewhl.vks; outlay=outvimg.layout
        # upload AU to bitstream buffer with slice offsets
        off=0; sliceoffs=UInt32[]
        for sl in au
            push!(sliceoffs,UInt32(off))
            unsafe_copyto!(bmap+off, pointer(vcat(UInt8[0,0,1],sl)), length(sl)+3); off+=length(sl)+3
        end
        # reference slots (active DPB)
        refpr=Ref{C.VkVideoPictureResourceInfoKHR}[]; refri=Ref{C.StdVideoDecodeH264ReferenceInfo}[]; refds=Ref{C.VkVideoDecodeH264DpbSlotInfoKHR}[]
        decref=C.VkVideoReferenceSlotInfoKHR[]; begref=C.VkVideoReferenceSlotInfoKHR[]
        for (slot,fn,pc0) in dpb
            rv=imgs[slot][2].vks
            pr=Ref(picres(rv)); ri=Ref(C.StdVideoDecodeH264ReferenceInfo(C.StdVideoDecodeH264ReferenceInfoFlags(pbytes(UInt32(0))),UInt16(fn),UInt16(0),(Int32(pc0),Int32(pc0))))
            ds=Ref(C.VkVideoDecodeH264DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,C_NULL,rp(ri)))
            push!(refpr,pr);push!(refri,ri);push!(refds,ds)
            push!(decref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,pc(ds),Int32(slot-1),rp(pr)))
            push!(begref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,C_NULL,Int32(slot-1),rp(pr)))
        end
        outpr=Ref(picres(outview))
        outri=Ref(C.StdVideoDecodeH264ReferenceInfo(C.StdVideoDecodeH264ReferenceInfoFlags(pbytes(UInt32(0))),UInt16(sh.fnum),UInt16(0),(Int32(poc),Int32(poc))))
        outds=Ref(C.VkVideoDecodeH264DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,C_NULL,rp(outri)))
        push!(begref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,C_NULL,Int32(-1),rp(outpr)))
        setup=Ref(C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,pc(outds),Int32(outslot-1),rp(outpr)))
        picflags=packflags((0, sh.stype==2 ? 1 : 0, sh.idr ? 1 : 0, 0, isref ? 1 : 0, 0))
        stdpic=Ref(C.StdVideoDecodeH264PictureInfo(C.StdVideoDecodeH264PictureInfoFlags(pbytes(picflags)),UInt8(0),UInt8(0),UInt8(0),UInt8(0),UInt16(sh.fnum),UInt16(sh.idrid),(Int32(poc),Int32(poc))))
        soff=copy(sliceoffs)
        h264pi=Ref(C.VkVideoDecodeH264PictureInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,C_NULL,rp(stdpic),UInt32(length(soff)),pointer(soff)))
        decinfo=Ref(C.VkVideoDecodeInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,pc(h264pi),UInt32(0),bbuf,UInt64(0),UInt64(cld(off,256)*256),outpr[],rp(setup),UInt32(length(decref)),isempty(decref) ? Ptr{C.VkVideoReferenceSlotInfoKHR}(C_NULL) : pointer(decref)))
        beginfo=Ref(C.VkVideoBeginCodingInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,C_NULL,UInt32(0),SESSION,PARAMS,UInt32(length(begref)),pointer(begref)))
        ctl=Ref(C.VkVideoCodingControlInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,C_NULL,UInt32(C.VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR)))
        endinfo=Ref(C.VkVideoEndCodingInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,C_NULL,UInt32(0)))
        bi=Ref(C.VkCommandBufferBeginInfo(C.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,C_NULL,UInt32(C.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT),Ptr{Cvoid}(C_NULL)))
        # This frame's luma lands in its own device-local LavaArray (cropped to
        # the display size DW×DH), copied out through the high-level VideoImage
        # `record_luma_copy!` — a real image→buffer transfer, no raw copy FFI.
        dst=Lava.LavaArray{UInt8,2}(undef,(DW,DH); extra_usage=UInt32(Vk.BUFFER_USAGE_TRANSFER_DST_BIT))
        dstbuf=dst.buf[]
        duv = chroma ? Lava.LavaArray{UInt8,2}(undef,(DW,DH÷2); extra_usage=UInt32(Vk.BUFFER_USAGE_TRANSFER_DST_BIT)) : nothing
        GC.@preserve PIN dst duv cbh outvimg refpr refri refds decref begref outpr outri outds setup stdpic soff h264pi decinfo beginfo ctl endinfo bi imgs begin
            ccall(dfp(w,"vkBeginCommandBuffer"),Int32,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(bi))
            for (slot,_,_) in dpb; lay=imgs[slot][1].layout; if lay[]!=DPBLAYOUT; emit(CB,barrier(imgs[slot][1].image.vks,lay[],DPBLAYOUT)); lay[]=DPBLAYOUT; end; end
            emit(CB,barrier(outimg,outlay[],DPBLAYOUT)); outlay[]=DPBLAYOUT
            ccall(dfp(w,"vkCmdBeginVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(beginfo))
            isfirst && ccall(dfp(w,"vkCmdControlVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(ctl))
            ccall(dfp(w,"vkCmdDecodeVideoKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(decinfo))
            ccall(dfp(w,"vkCmdEndVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(endinfo))
            emit(CB,barrier(outimg,DPBLAYOUT,TSRC)); outlay[]=TSRC   # outvimg.layout now TRANSFER_SRC
            Lava.record_luma_copy!(cbh, outvimg, dstbuf.buffer, dstbuf.pool_offset + dst.offset)
            if chroma
                uvbuf = duv.buf[]
                Lava.record_chroma_copy!(cbh, outvimg, uvbuf.buffer, uvbuf.pool_offset + duv.offset)
            end
            ccall(dfp(w,"vkEndCommandBuffer"),Int32,(C.VkCommandBuffer,),CB)
        end
        cbref=Ref(CB); si=Ref(C.VkSubmitInfo(C.VK_STRUCTURE_TYPE_SUBMIT_INFO,C_NULL,UInt32(0),Ptr{C.VkSemaphore}(C_NULL),Ptr{UInt32}(C_NULL),UInt32(1),Base.unsafe_convert(Ptr{C.VkCommandBuffer},cbref),UInt32(0),Ptr{C.VkSemaphore}(C_NULL)))
        GC.@preserve si cbref ccall(dfp(w,"vkQueueSubmit"),Int32,(Ptr{Cvoid},UInt32,Ptr{Cvoid},Ptr{Cvoid}),w.q.vks,UInt32(1),Base.unsafe_convert(Ptr{Cvoid},si),C_NULL)
        ccall(dfp(w,"vkQueueWaitIdle"),Int32,(Ptr{Cvoid},),w.q.vks)
        push!(outputs,(gop,poc,dst,duv))      # GPU-resident luma (+chroma), cropped
        if isref
            adaptive, mmops = parse_mmco(au[1], sps, pps)
            if adaptive
                # Adaptive marking: MMCO fully controls the DPB (no sliding window).
                MaxFN = 1<<(sps.log2fn+4)
                picnum(fn) = fn > sh.fnum ? fn - MaxFN : fn   # PicNum = FrameNumWrap (frames)
                for (op,arg,_lt) in mmops
                    if op==1                              # unmark a short-term ref by PicNum
                        px = sh.fnum - (arg+1)
                        k = findfirst(e -> picnum(e[2]) == px, dpb)
                        k !== nothing && (push!(freeslots, dpb[k][1]); deleteat!(dpb, k))
                    elseif op==5                          # unmark all references
                        for e in dpb; push!(freeslots, e[1]); end; empty!(dpb)
                    end
                    # ops 2/3/4/6 (long-term refs) unused by the encoders we target
                end
                push!(dpb,(outslot,sh.fnum,poc))
            else
                push!(dpb,(outslot,sh.fnum,poc))
                while length(dpb)>sps.maxref; old=popfirst!(dpb); push!(freeslots,old[1]); end
            end
        else; push!(freeslots,outslot); end
        isfirst=false
    end
    # Display order is (GOP, POC): pic_order_cnt resets to 0 at every IDR, so
    # POC alone interleaves frames from different coded video sequences.
    sort!(outputs, by=x->(x[1], x[2]))
    ys  = Lava.LavaArray{UInt8,2}[o[3] for o in outputs]
    uvs = chroma ? Lava.LavaArray{UInt8,2}[o[4] for o in outputs] : Lava.LavaArray{UInt8,2}[]
    (DW, DH, ys, uvs)
end
end # module
