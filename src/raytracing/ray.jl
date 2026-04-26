"""
    Ray(origin::Point3f, direction::Vec3f, tmin::Float32, tmax::Float32)

A ray for inline ray-query intrinsics. `direction` need not be unit-length;
it is passed verbatim to `OpRayQueryInitializeKHR`.
"""
struct Ray
    origin::Point3f
    direction::Vec3f
    tmin::Float32
    tmax::Float32
end
