// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "onnxruntime_c_api.h"
#include "onnxruntime_training_c_api.h"
#include "onnxruntime_training_cxx_api.h"

#include "orttraining/training_api/checkpoint.h"

namespace onnxruntime::training::test {

#define MODEL_FOLDER ORT_TSTR("testdata/training_api/")

TEST(TrainingCApiTest, GetState) {
  auto model_uri = MODEL_FOLDER "training_model.onnx";

  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");
  Ort::TrainingSession training_session = Ort::TrainingSession(Ort::SessionOptions(), checkpoint_state, model_uri);

  Ort::CheckpointState state = training_session.GetState(false);

  // Check if the retrieved state and the original checkpoint state are equal
  OrtCheckpointState* checkpoint_state_c = checkpoint_state;
  OrtCheckpointState* state_c = state;
  auto checkpoint_state_internal = reinterpret_cast<onnxruntime::training::api::CheckpointState*>(checkpoint_state_c);
  auto state_internal = reinterpret_cast<onnxruntime::training::api::CheckpointState*>(state_c);

  ASSERT_FALSE(checkpoint_state_internal->module_checkpoint_state.named_parameters.empty());
  ASSERT_EQ(checkpoint_state_internal->module_checkpoint_state.named_parameters.size(),
            state_internal->module_checkpoint_state.named_parameters.size());
  for (auto& [key, value] : checkpoint_state_internal->module_checkpoint_state.named_parameters) {
    ASSERT_TRUE(state_internal->module_checkpoint_state.named_parameters.count(key));
  }
}

TEST(TrainingCApiTest, AddIntProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  int64_t value = 365 * 24;

  checkpoint_state.AddProperty("hours in a year", value);

  auto property = checkpoint_state.GetProperty("hours in a year");

  ASSERT_EQ(std::get<int64_t>(property), value);
}

TEST(TrainingCApiTest, AddFloatProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  float value = 3.14f;

  checkpoint_state.AddProperty("pi", value);

  auto property = checkpoint_state.GetProperty("pi");

  ASSERT_EQ(std::get<float>(property), value);
}

TEST(TrainingCApiTest, AddStringProperty) {
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(MODEL_FOLDER "checkpoint.ckpt");

  std::string value("onnxruntime");

  checkpoint_state.AddProperty("framework", value);

  auto property = checkpoint_state.GetProperty("framework");

  ASSERT_EQ(std::get<std::string>(property), value);
}

}  // namespace onnxruntime::training::test
