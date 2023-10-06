#include "camera.cuh"
#include "camera_info.cuh"
#include "camera_utils.cuh"
#include "parameters.cuh"
#include <string>
#include <torch/torch.h>
#include <utility>

Camera::Camera(int imported_colmap_id,
               Eigen::Matrix3f R, Eigen::Vector3f T,
               float FoVx, float FoVy,
               torch::Tensor image,
               std::string image_name,
               int uid,
               float scale) : _colmap_id(imported_colmap_id),
                              _R(R),
                              _T(T),
                              _FoVx(FoVx),
                              _FoVy(FoVy),
                              _image_name(std::move(std::move(image_name))),
                              _uid(uid),
                              _scale(scale) {

    _original_image = torch::clamp(image, 0.f, 1.f);
    _image_width = _original_image.size(2);
    _image_height = _original_image.size(1);

    _zfar = 100.f;
    _znear = 0.01f;

    _world_view_transform = getWorld2View2(R, T, Eigen::Vector3f::Zero(), _scale).to(torch::kCUDA, true);
    //std::cout<< "world_view_transform: " << _world_view_transform <<std::endl;
    _projection_matrix = getProjectionMatrix(_znear, _zfar, _FoVx, _FoVy).to(torch::kCUDA, true);
    _full_proj_transform = _world_view_transform.unsqueeze(0).bmm(_projection_matrix.unsqueeze(0)).squeeze(0);
    _camera_center = _world_view_transform.inverse()[3].slice(0, 0, 3);
}

// TODO: I have skipped the resolution for now.
Camera loadCam(const gs::param::ModelParameters& params, int id, CameraInfo& cam_info) {
    // Create a torch::Tensor from the image data
    torch::Tensor original_image_tensor = torch::from_blob(cam_info._img_data,
                                                           {cam_info._img_h, cam_info._img_w, cam_info._channels},        // img size
                                                           {cam_info._img_w * cam_info._channels, cam_info._channels, 1}, // stride
                                                           torch::kUInt8);
    original_image_tensor = original_image_tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;

    free_image(cam_info._img_data); // we dont longer need the image here.
    cam_info._img_data = nullptr;   // Assure that we dont use the image data anymore.

    return Camera(cam_info._camera_ID, cam_info._R, cam_info._T, cam_info._fov_x, cam_info._fov_y, original_image_tensor,
                  cam_info._image_name, id);
}