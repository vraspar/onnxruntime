// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_training_c_api.h"
#include <optional>
#include <variant>

namespace Ort::detail {

#define ORT_DECLARE_TRAINING_RELEASE(NAME) \
  void OrtRelease(Ort##NAME* ptr);

// These release methods must be forward declared before including onnxruntime_cxx_api.h
// otherwise class Base won't be aware of them
ORT_DECLARE_TRAINING_RELEASE(CheckpointState);
ORT_DECLARE_TRAINING_RELEASE(TrainingSession);

}  // namespace Ort::detail

#include "onnxruntime_cxx_api.h"

namespace Ort {

inline const OrtTrainingApi& GetTrainingApi() { return *GetApi().GetTrainingApi(ORT_API_VERSION); }

namespace detail {

#define ORT_DEFINE_TRAINING_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { GetTrainingApi().Release##NAME(ptr); }

ORT_DEFINE_TRAINING_RELEASE(CheckpointState);
ORT_DEFINE_TRAINING_RELEASE(TrainingSession);

#undef ORT_DECLARE_TRAINING_RELEASE
#undef ORT_DEFINE_TRAINING_RELEASE

}  // namespace detail

using Property = std::variant<int64_t, float, std::string>;

class CheckpointState : public detail::Base<OrtCheckpointState> {
 public:
  CheckpointState(OrtCheckpointState* checkpoint_state) { p_ = checkpoint_state; }

  CheckpointState() = delete;

  /** \brief Loads the checkpoint at provided path and returns the checkpoint state
   *
   * Wraps OrtTrainingApi::LoadCheckpoint
   *
   */
  static CheckpointState LoadCheckpoint(const std::basic_string<ORTCHAR_T>& path_to_checkpoint);

  /** \brief Saves the state of the training session to a checkpoint file provided by the given path.
   *
   * Wraps OrtTrainingApi::SaveCheckpoint
   *
   */
  static void SaveCheckpoint(const CheckpointState& checkpoint_state,
                             const std::basic_string<ORTCHAR_T>& path_to_checkpoint);

  /** \brief Adds the given property to the state.
   *
   * Wraps OrtTrainingApi::AddProperty
   *
   */
  void AddProperty(const std::basic_string<ORTCHAR_T>& property_name, const Property& property_value);

  /** \brief Gets the property associated with the given name from the state.
   *
   * Wraps OrtTrainingApi::GetProperty
   *
   */
  Property GetProperty(const std::basic_string<ORTCHAR_T>& property_name);
};

/** \brief Manage the training loop using this class
 *
 * Wraps OrtTrainingSession
 *
 */
class TrainingSession : public detail::Base<OrtTrainingSession> {
 private:
  size_t training_model_output_count_, eval_model_output_count_;

 public:
  TrainingSession(const SessionOptions& session_options, CheckpointState& checkpoint_state,
                  const std::basic_string<ORTCHAR_T>& train_model_path,
                  const std::optional<std::basic_string<ORTCHAR_T>>& eval_model_path = std::nullopt,
                  const std::optional<std::basic_string<ORTCHAR_T>>& optimizer_model_path = std::nullopt);

  /** \brief Run the train step returning results in an Ort allocated vector.
   *
   * Wraps OrtTrainingApi::TrainStep
   *
   * \param[in] input_values Array of Value objects in the order expected by the training model.
   * \return A std::vector of Value objects that represents the output of the forward pass.
   */
  std::vector<Value> TrainStep(const std::vector<Value>& input_values);

  /** \brief Lazily resets the gradients of the trainable parameters.
   *
   * Wraps OrtTrainingApi::LazyResetGrad
   *
   */
  void LazyResetGrad();

  /** \brief Run the evaluation step returning results in an Ort allocated vector.
   *
   * Wraps OrtTrainingApi::EvalStep
   *
   * \param[in] input_values Array of Value objects in the order expected by the eval model.
   * \return A std::vector of Value objects that represents the output of the eval pass.
   */
  std::vector<Value> EvalStep(const std::vector<Value>& input_values);

  /** \brief Set the learning rate to be used by the optimizer for parameter updates.
   *
   * Wraps OrtTrainingApi::SetLearningRate
   *
   * \param[in] learning_rate float value representing the constant learning rate to be used.
   */
  void SetLearningRate(float learning_rate);

  /** \brief Get the current learning rate that is being used by the optimizer.
   *
   * Wraps OrtTrainingApi::GetLearningRate
   *
   * \return float representing the current learning rate.
   */
  float GetLearningRate() const;

  /** \brief Register the linear learning rate scheduler for the training session.
   *
   * Wraps OrtTrainingApi::RegisterLinearLRScheduler
   *
   * \param[in] warmup_step_count Number of steps in the warmup phase.
   * \param[in] total_step_count Total number of training steps.
   * \param[in] initial_lr Initial learning rate to use.
   */
  void RegisterLinearLRScheduler(int64_t warmup_step_count, int64_t total_step_count,
                                 float initial_lr);

  /** \brief Updates the learning rate based on the lr scheduler.
   *
   * Wraps OrtTrainingApi::SchedulerStep
   *
   */
  void SchedulerStep();

  /** \brief Runs the optimizer model and updates the model parameters.
   *
   * Wraps OrtTrainingApi::OptimizerStep
   *
   */
  void OptimizerStep();

  /** \brief Exports a model that can be used for inferencing with the inference session.
   *
   * Wraps OrtTrainingApi::ExportModelForInferencing
   *
   */
  void ExportModelForInferencing(const std::basic_string<ORTCHAR_T>& inference_model_path,
                                 const std::vector<std::string>& graph_output_names);

  /** \brief Gets the current training state of the session.
   *
   * Wraps OrtTrainingApi::GetState
   *
   */
  CheckpointState GetState(const bool include_optimizer_state);
};

/** \brief Sets the given seed for random number generation.
 *
 * Wraps OrtTrainingApi::SetSeed
 *
 */
void SetSeed(const int64_t seed);

}  // namespace Ort

#include "onnxruntime_training_cxx_inline.h"
