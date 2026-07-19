# Hardware H.264 decode on the GPU via VK_KHR_video_decode_queue.
# Input: an H.264 Annex-B elementary stream (SPS/PPS/slice NAL units).
# Output: luma (Y) planes in display order. Everything (session, DPB reference
# management, POC ordering) is driven through VulkanCore's raw video-decode FFI —
# Vulkan.jl's high-level wrapper does not generate the video API. A minimal
# exp-Golomb parser reads the SPS/PPS and per-slice headers to build the
# StdVideoH264* structs and manage the decoded-picture buffer.
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
    decode_h264_luma(annexb::Vector{UInt8}; maxframes=typemax(Int)) -> (w, h, Vector{Matrix{UInt8}})

Hardware-decode an H.264 Annex-B stream on the GPU and return the luma (Y) planes
in DISPLAY order (cropped to the display size). Grayscale = exactly the NV12 Y plane.
"""
function decode_h264_luma(ctx, annexb::Vector{UInt8}; maxframes::Int=typemax(Int))
    ctx.video_decode_available || error("device has no video decode support")
    w=Ctxwrap(ctx, ctx.device, UInt32(ctx.video_decode_queue_family_index), ctx.video_decode_queue)
    dev=w.dev
    nalu=split_nals(annexb)
    sps = parse_sps(first(n[3] for n in nalu if n[1]==7))
    pps = parse_pps(first(n[3] for n in nalu if n[1]==8))
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
    # DPB image pool
    nslots=Int(maxslots)
    idsw=C.VkComponentSwizzle(0)
    imgs=Vector{Any}(undef,nslots)  # (img, view, layout::Ref{UInt32})
    for k in 1:nslots
        u=UInt32(C.VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)|UInt32(C.VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR)|UInt32(C.VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        ci=Ref(C.VkImageCreateInfo(C.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,Ptr{Cvoid}(rp(pl)),UInt32(0),C.VK_IMAGE_TYPE_2D,fmt,C.VkExtent3D(CW,CH,1),UInt32(1),UInt32(1),C.VK_SAMPLE_COUNT_1_BIT,C.VK_IMAGE_TILING_OPTIMAL,u,C.VK_SHARING_MODE_EXCLUSIVE,UInt32(0),Ptr{UInt32}(C_NULL),C.VK_IMAGE_LAYOUT_UNDEFINED))
        ri=Ref{C.VkImage}(); GC.@preserve ci pl PIN ccall(dfp(w,"vkCreateImage"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(ci),C_NULL,rp(ri)); im=ri[]
        mr=Ref(C.VkMemoryRequirements(UInt64(0),UInt64(0),UInt32(0))); ccall(dfp(w,"vkGetImageMemoryRequirements"),Cvoid,(Ptr{Cvoid},C.VkImage,Ptr{Cvoid}),dev.vks,im,rp(mr))
        m=Vk.unwrap(Vk.allocate_memory(dev,mr[].size,memtype(w,mr[].memoryTypeBits,C.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))); pin(m)
        ccall(dfp(w,"vkBindImageMemory"),Int32,(Ptr{Cvoid},C.VkImage,C.VkDeviceMemory,UInt64),dev.vks,im,m.vks,UInt64(0))
        vci=Ref(C.VkImageViewCreateInfo(C.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,C_NULL,UInt32(0),im,C.VK_IMAGE_VIEW_TYPE_2D,fmt,C.VkComponentMapping(idsw,idsw,idsw,idsw),C.VkImageSubresourceRange(UInt32(C.VK_IMAGE_ASPECT_COLOR_BIT),UInt32(0),UInt32(1),UInt32(0),UInt32(1))))
        rv=Ref{C.VkImageView}(); GC.@preserve vci ccall(dfp(w,"vkCreateImageView"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(vci),C_NULL,rp(rv))
        imgs[k]=(im,rv[],Ref(UInt32(C.VK_IMAGE_LAYOUT_UNDEFINED)))
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
    rbsz=UInt64(CW*CH*3÷2)
    rbuf,rmem,rmap = let
        ci=Ref(C.VkBufferCreateInfo(C.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,C_NULL,UInt32(0),rbsz,UInt32(C.VK_BUFFER_USAGE_TRANSFER_DST_BIT),C.VK_SHARING_MODE_EXCLUSIVE,UInt32(0),Ptr{UInt32}(C_NULL)))
        rb=Ref{C.VkBuffer}(); GC.@preserve ci ccall(dfp(w,"vkCreateBuffer"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(ci),C_NULL,rp(rb)); bf=rb[]
        mr=Ref(C.VkMemoryRequirements(UInt64(0),UInt64(0),UInt32(0))); ccall(dfp(w,"vkGetBufferMemoryRequirements"),Cvoid,(Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}),dev.vks,bf,rp(mr))
        m=Vk.unwrap(Vk.allocate_memory(dev,mr[].size,memtype(w,mr[].memoryTypeBits,UInt32(C.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)|UInt32(C.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)))); pin(m)
        ccall(dfp(w,"vkBindBufferMemory"),Int32,(Ptr{Cvoid},C.VkBuffer,C.VkDeviceMemory,UInt64),dev.vks,bf,m.vks,UInt64(0))
        pmap=Ref{Ptr{Cvoid}}(); ccall(dfp(w,"vkMapMemory"),Int32,(Ptr{Cvoid},C.VkDeviceMemory,UInt64,UInt64,UInt32,Ptr{Cvoid}),dev.vks,m.vks,UInt64(0),rbsz,UInt32(0),Base.unsafe_convert(Ptr{Ptr{Cvoid}},pmap))
        (bf,m,Ptr{UInt8}(pmap[]))
    end
    # command pool + buffer
    cp=Ref(C.VkCommandPoolCreateInfo(C.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,C_NULL,UInt32(C.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT),w.qf))
    rpool=Ref{C.VkCommandPool}(); GC.@preserve cp ccall(dfp(w,"vkCreateCommandPool"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(cp),C_NULL,rp(rpool)); POOL=rpool[]
    ai=Ref(C.VkCommandBufferAllocateInfo(C.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,C_NULL,POOL,C.VK_COMMAND_BUFFER_LEVEL_PRIMARY,UInt32(1)))
    cbv=Vector{C.VkCommandBuffer}(undef,1); GC.@preserve ai ccall(dfp(w,"vkAllocateCommandBuffers"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(ai),pointer(cbv)); CB=cbv[1]

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
    prevmsb=0; prevlsb=0; maxpoclsb=1<<(sps.log2poc+4)
    outputs=Tuple{Int,Matrix{UInt8}}[]   # (poc, luma)
    isfirst=true
    for (ai_i,au) in enumerate(aus)
        ai_i>maxframes && break
        sh=parse_slice(au[1],sps)
        if sh.idr
            for (slot,_,_) in dpb; push!(freeslots,slot); end; empty!(dpb); prevmsb=0; prevlsb=0
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
        outimg,outview,outlay=imgs[outslot]
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
            _,rv,_=imgs[slot]
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
        p0(a)=C.VkImageSubresourceLayers(UInt32(a),UInt32(0),UInt32(0),UInt32(1))
        regs=[C.VkBufferImageCopy(UInt64(0),UInt32(0),UInt32(0),p0(C.VK_IMAGE_ASPECT_PLANE_0_BIT),C.VkOffset3D(0,0,0),C.VkExtent3D(CW,CH,1)),
              C.VkBufferImageCopy(UInt64(CW*CH),UInt32(0),UInt32(0),p0(C.VK_IMAGE_ASPECT_PLANE_1_BIT),C.VkOffset3D(0,0,0),C.VkExtent3D(CW÷2,CH÷2,1))]
        GC.@preserve PIN refpr refri refds decref begref outpr outri outds setup stdpic soff h264pi decinfo beginfo ctl endinfo bi regs imgs begin
            ccall(dfp(w,"vkBeginCommandBuffer"),Int32,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(bi))
            for (slot,_,_) in dpb; _,_,lay=imgs[slot]; if lay[]!=DPBLAYOUT; emit(CB,barrier(imgs[slot][1],lay[],DPBLAYOUT)); lay[]=DPBLAYOUT; end; end
            emit(CB,barrier(outimg,outlay[],DPBLAYOUT)); outlay[]=DPBLAYOUT
            ccall(dfp(w,"vkCmdBeginVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(beginfo))
            isfirst && ccall(dfp(w,"vkCmdControlVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(ctl))
            ccall(dfp(w,"vkCmdDecodeVideoKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(decinfo))
            ccall(dfp(w,"vkCmdEndVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(endinfo))
            emit(CB,barrier(outimg,DPBLAYOUT,TSRC)); outlay[]=TSRC
            ccall(dfp(w,"vkCmdCopyImageToBuffer"),Cvoid,(C.VkCommandBuffer,C.VkImage,UInt32,C.VkBuffer,UInt32,Ptr{Cvoid}),CB,outimg,TSRC,rbuf,UInt32(2),pointer(regs))
            ccall(dfp(w,"vkEndCommandBuffer"),Int32,(C.VkCommandBuffer,),CB)
        end
        cbh=Ref(CB); si=Ref(C.VkSubmitInfo(C.VK_STRUCTURE_TYPE_SUBMIT_INFO,C_NULL,UInt32(0),Ptr{C.VkSemaphore}(C_NULL),Ptr{UInt32}(C_NULL),UInt32(1),Base.unsafe_convert(Ptr{C.VkCommandBuffer},cbh),UInt32(0),Ptr{C.VkSemaphore}(C_NULL)))
        GC.@preserve si cbh ccall(dfp(w,"vkQueueSubmit"),Int32,(Ptr{Cvoid},UInt32,Ptr{Cvoid},Ptr{Cvoid}),w.q.vks,UInt32(1),Base.unsafe_convert(Ptr{Cvoid},si),C_NULL)
        ccall(dfp(w,"vkQueueWaitIdle"),Int32,(Ptr{Cvoid},),w.q.vks)
        # read back Y plane, crop to display
        luma=Matrix{UInt8}(undef,DW,DH)
        if DW==CW
            unsafe_copyto!(pointer(luma), rmap, DW*DH)              # Y plane contiguous
        else
            for y in 1:DH; unsafe_copyto!(pointer(luma,(y-1)*DW+1), rmap+(y-1)*CW, DW); end
        end
        push!(outputs,(poc,luma))
        if isref
            push!(dpb,(outslot,sh.fnum,poc))
            while length(dpb)>sps.maxref; old=popfirst!(dpb); push!(freeslots,old[1]); end
        else; push!(freeslots,outslot); end
        isfirst=false
    end
    sort!(outputs, by=x->x[1])
    (DW,DH,[o[2] for o in outputs])
end
end # module
