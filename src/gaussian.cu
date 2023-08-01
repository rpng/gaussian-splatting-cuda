#include "gaussian.cuh"

GaussianModel::GaussianModel(int sh_degree) : max_sh_degree(sh_degree),
                                              active_sh_degree(0),
                                              _xyz_scheduler_args(Expon_lr_func(0.0, 1.0)) {

    // Assuming these are 1D tensors
    _xyz = torch::empty({0});
    _features_dc = torch::empty({0});
    _features_rest = torch::empty({0});
    _scaling = torch::empty({0});
    _rotation = torch::empty({0});
    _opacity = torch::empty({0});
    _max_radii2D = torch::empty({0});
    _xyz_gradient_accum = torch::empty({0});
    optimizer = nullptr;

    register_parameter("xyz", _xyz, true);
    register_parameter("features_dc", _features_dc, true);
    register_parameter("features_rest", _features_rest, true);
    register_parameter("scaling", _scaling, true);
    register_parameter("rotation", _rotation, true);
    register_parameter("opacity", _opacity, true);

    // Scaling activation and its inverse
    _scaling_activation = torch::exp;
    _scaling_inverse_activation = torch::log;

    // Covariance activation function
    _covariance_activation = [](const torch::Tensor& scaling, const torch::Tensor& scaling_modifier, const torch::Tensor& rotation) {
        auto L = build_scaling_rotation(scaling_modifier * scaling, rotation);
        auto actual_covariance = torch::mm(L, L.transpose(1, 2));
        auto symm = strip_symmetric(actual_covariance);
        return symm;
    };

    // Opacity activation and its inverse
    _opacity_activation = torch::sigmoid;
    _inverse_opacity_activation = inverse_sigmoid;

    // Rotation activation function
    _rotation_activation = torch::nn::functional::normalize;
}

/**
 * @brief Fetches the features of the Gaussian model
 *
 * This function concatenates _features_dc and _features_rest along the second dimension.
 *
 * @return Tensor of the concatenated features
 */
torch::Tensor GaussianModel::get_features() const {
    auto features_dc = _features_dc;
    auto features_rest = _features_rest;
    return torch::cat({features_dc, features_rest}, 1);
}

/**
 * @brief Increment the SH degree by 1
 *
 * This function increments the active_sh_degree by 1, up to a maximum of max_sh_degree.
 */
void GaussianModel::oneupSHdegree() {
    if (active_sh_degree < max_sh_degree) {
        active_sh_degree++;
    }
}

/**
 * @brief Initialize Gaussian Model from a Point Cloud.
 *
 * This function creates a Gaussian model from a given PointCloud object. It also sets
 * the spatial learning rate scale. The model's features, scales, rotations, and opacities
 * are initialized based on the input point cloud.
 *
 * @param pcd The input point cloud
 * @param spatial_lr_scale The spatial learning rate scale
 */
void GaussianModel::create_from_pcd(PointCloud& pcd, float spatial_lr_scale) {
    _spatial_lr_scale = spatial_lr_scale;

    torch::Tensor fused_point_cloud = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size())}).to(torch::kCUDA);
    auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size())}).to(torch::kCUDA));

    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3), torch::indexing::Slice(1)}, 0.0);

    std::cout << "Number of points at initialisation : " << fused_point_cloud.size(0) << std::endl;

    auto dist2 = torch::clamp_min(distCUDA2(torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size())}).to(torch::kCUDA)), 0.0000001);
    auto scales = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3});
    auto rots = torch::zeros({fused_point_cloud.size(0), 4}).to(torch::kCUDA);
    rots.index_put_({torch::indexing::Slice(), 0}, 1);

    auto opacities = inverse_sigmoid(0.5 * torch::ones({fused_point_cloud.size(0), 1}).to(torch::kCUDA));

    _xyz = fused_point_cloud.set_requires_grad(true);
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).transpose(1, 2).contiguous();
    _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1)}).transpose(1, 2).contiguous();
    _scaling = scales;
    _rotation = rots;
    _opacity = opacities;
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
}

/**
 * @brief Setup the Gaussian Model for training
 *
 * This function sets up the Gaussian model for training by initializing several
 * parameters and settings based on the provided OptimizationParameters object.
 *
 * @param params The OptimizationParameters object providing the settings for training
 */
void GaussianModel::training_setup(const OptimizationParameters& params) {
    this->percent_dense = params.percent_dense;
    this->_xyz_gradient_accum = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_denom = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);

    register_parameter("xyz", this->_xyz);
    optimizer = std::make_unique<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(0.0).eps(1e-15));
    this->_xyz_scheduler_args = Expon_lr_func(params.position_lr_init * this->_spatial_lr_scale,
                                              params.position_lr_final * this->_spatial_lr_scale,
                                              params.position_lr_delay_mult,
                                              params.position_lr_max_steps);
}
