// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_training_c_api.h"
#include "onnxruntime_cxx_api.h"

namespace Ort {

inline TrainingSession::TrainingSession(const SessionOptions& session_options,
                                        CheckpointState& checkpoint_state,
                                        const std::basic_string<ORTCHAR_T>& train_model_path,
                                        const std::optional<std::basic_string<ORTCHAR_T>>& eval_model_path,
                                        const std::optional<std::basic_string<ORTCHAR_T>>& optimizer_model_path) {
  Env env = Env();
  ThrowOnError(GetTrainingApi().CreateTrainingSession(
      env, session_options, checkpoint_state,
      train_model_path.c_str(),
      eval_model_path.has_value() ? eval_model_path.value().c_str() : nullptr,
      optimizer_model_path.has_value() ? optimizer_model_path.value().c_str() : nullptr,
      &p_));

  ThrowOnError(GetTrainingApi().TrainingSessionGetTrainingModelOutputCount(p_, &training_model_output_count_));

  ThrowOnError(GetTrainingApi().TrainingSessionGetEvalModelOutputCount(p_, &eval_model_output_count_));
}

inline std::vector<Value> TrainingSession::TrainStep(const std::vector<Value>& input_values) {
  std::vector<Value> output_values;
  output_values.reserve(training_model_output_count_);
  for (size_t i = 0; i < training_model_output_count_; i++) output_values.emplace_back(nullptr);
  auto ort_input_values = reinterpret_cast<const OrtValue* const*>(input_values.data());
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values.data());
  RunOptions run_options;
  ThrowOnError(GetTrainingApi().TrainStep(
      p_, run_options, input_values.size(), ort_input_values,
      training_model_output_count_, ort_output_values));

  return output_values;
}

inline void TrainingSession::LazyResetGrad() {
  ThrowOnError(GetTrainingApi().LazyResetGrad(p_));
}

inline std::vector<Value> TrainingSession::EvalStep(const std::vector<Value>& input_values) {
  std::vector<Value> output_values;
  output_values.reserve(eval_model_output_count_);
  for (size_t i = 0; i < eval_model_output_count_; i++) output_values.emplace_back(nullptr);
  auto ort_input_values = reinterpret_cast<const OrtValue* const*>(input_values.data());
  auto ort_output_values = reinterpret_cast<OrtValue**>(output_values.data());
  RunOptions run_options;
  ThrowOnError(GetTrainingApi().EvalStep(
      p_, run_options, input_values.size(), ort_input_values,
      training_model_output_count_, ort_output_values));

  return output_values;
}

inline void TrainingSession::SetLearningRate(float learning_rate) {
  ThrowOnError(GetTrainingApi().SetLearningRate(p_, learning_rate));
}

inline float TrainingSession::GetLearningRate() const {
  float learning_rate = 0;
  ThrowOnError(GetTrainingApi().GetLearningRate(p_, &learning_rate));
  return learning_rate;
}

inline void TrainingSession::RegisterLinearLRScheduler(int64_t warmup_step_count, int64_t total_step_count,
                                                       float initial_lr) {
  ThrowOnError(GetTrainingApi().RegisterLinearLRScheduler(p_, warmup_step_count, total_step_count,
                                                          initial_lr));
}

inline void TrainingSession::SchedulerStep() {
  ThrowOnError(GetTrainingApi().SchedulerStep(p_));
}

inline void TrainingSession::OptimizerStep() {
  RunOptions run_options;
  ThrowOnError(GetTrainingApi().OptimizerStep(p_, run_options));
}

inline CheckpointState TrainingSession::GetState(const bool include_optimizer_state) {
  OrtCheckpointState* state;
  ThrowOnError(GetTrainingApi().GetState(p_, include_optimizer_state, &state));
  return CheckpointState(state);
}

inline CheckpointState CheckpointState::LoadCheckpoint(const std::basic_string<ORTCHAR_T>& path_to_checkpoint) {
  OrtCheckpointState* checkpoint_state;
  ThrowOnError(GetTrainingApi().LoadCheckpoint(path_to_checkpoint.c_str(), &checkpoint_state));
  return CheckpointState(checkpoint_state);
}

inline void CheckpointState::SaveCheckpoint(const CheckpointState& checkpoint_states,
                                            const std::basic_string<ORTCHAR_T>& path_to_checkpoint) {
  ThrowOnError(GetTrainingApi().SaveCheckpoint(checkpoint_states, path_to_checkpoint.c_str()));
}

inline void TrainingSession::ExportModelForInferencing(const std::basic_string<ORTCHAR_T>& inference_model_path,
                                                       const std::vector<std::string>& graph_output_names) {
  std::vector<const char*> output_names;
  output_names.reserve(graph_output_names.size());
  for (const auto& output_name : graph_output_names) {
    output_names.push_back(output_name.c_str());
  }
  ThrowOnError(GetTrainingApi().ExportModelForInferencing(
      p_, inference_model_path.c_str(), graph_output_names.size(), output_names.data()));
}

inline void SetSeed(const int64_t seed) {
  ThrowOnError(GetTrainingApi().SetSeed(seed));
}

inline void CheckpointState::AddProperty(const std::basic_string<ORTCHAR_T>& property_name,
                                         const Property& property_value) {
  if (std::holds_alternative<int64_t>(property_value)) {
    int64_t value = std::get<int64_t>(property_value);
    void* value_p = &value;
    ThrowOnError(GetTrainingApi().AddProperty(p_, property_name.c_str(), OrtPropertyType::OrtIntProperty, value_p));
  } else if (std::holds_alternative<float>(property_value)) {
    float value = std::get<float>(property_value);
    void* value_p = &value;
    ThrowOnError(GetTrainingApi().AddProperty(p_, property_name.c_str(), OrtPropertyType::OrtFloatProperty, value_p));
  } else if (std::holds_alternative<std::string>(property_value)) {
    std::string value = std::get<std::string>(property_value);
    auto buffer = std::make_unique<char[]>(value.length() + 1).release();
    memcpy(buffer, value.c_str(), value.length());
    ThrowOnError(GetTrainingApi().AddProperty(p_, property_name.c_str(), OrtPropertyType::OrtStringProperty, buffer));
  } else {
    ThrowStatus(Status("Unknown property type received.", OrtErrorCode::ORT_INVALID_ARGUMENT));
  }
}

inline Property CheckpointState::GetProperty(const std::basic_string<ORTCHAR_T>& property_name) {
  void* property_value = nullptr;
  OrtPropertyType property_type;

  ThrowOnError(GetTrainingApi().GetProperty(p_, property_name.c_str(), &property_type, &property_value));

  Property property;

  switch (property_type) {
    case OrtPropertyType::OrtIntProperty: {
      auto value_p = reinterpret_cast<int64_t*>(property_value);
      property = *value_p;
      delete value_p;
      break;
    }
    case OrtPropertyType::OrtFloatProperty: {
      auto value_p = reinterpret_cast<float*>(property_value);
      property = *value_p;
      delete value_p;
      break;
    }
    case OrtPropertyType::OrtStringProperty: {
      auto value_p = reinterpret_cast<char*>(property_value);
      property = std::string(value_p);
      delete value_p;
      break;
    }
    default: {
      ThrowStatus(Status("Unknown property type received.", OrtErrorCode::ORT_INVALID_ARGUMENT));
      break;
    }
  }

  return property;
}

}  // namespace Ort
