#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Tests for FAISSx Server Vector Transformations
# https://github.com/muxi-ai/faissx
#
# Copyright (C) 2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for FAISSx Server Vector Transformations

This module contains tests for the vector transformations functionality
in the FAISSx server.
"""

import unittest
import numpy as np
import faiss
from faissx.server.transformations import (
    parse_transform_type, create_transformation, create_pretransform_index,
    is_transform_trained, get_transform_training_requirements, train_transform
)


class TestTransformations(unittest.TestCase):
    """Tests for the transformations module."""

    def test_parse_transform_type(self):
        """Test parsing compound index types."""
        # Test PCA transform
        transform_type, base_type, params = parse_transform_type("PCA32,L2")
        self.assertEqual(transform_type, "PCA")
        self.assertEqual(base_type, "L2")
        self.assertEqual(params.get("output_dim"), 32)

        # Test L2 normalization
        transform_type, base_type, params = parse_transform_type("NORM,HNSW32")
        self.assertEqual(transform_type, "NORM")
        self.assertEqual(base_type, "HNSW32")
        self.assertEqual(params, {})

        # Test OPQ transform
        transform_type, base_type, params = parse_transform_type("OPQ8_32,IVF100,PQ8")
        self.assertEqual(transform_type, "OPQ")
        self.assertEqual(base_type, "IVF100,PQ8")
        self.assertEqual(params.get("M"), 8)
        self.assertEqual(params.get("output_dim"), 32)

        # Test no transform
        transform_type, base_type, params = parse_transform_type("L2")
        self.assertIsNone(transform_type)
        self.assertEqual(base_type, "L2")
        self.assertEqual(params, {})

    def test_create_transformation(self):
        """Test creating different transformation types."""
        # Test PCA transformation
        input_dim = 128
        output_dim = 64
        transform, info = create_transformation("PCA", input_dim, output_dim)

        self.assertIsInstance(transform, faiss.PCAMatrix)
        self.assertEqual(transform.d_in, input_dim)
        self.assertEqual(transform.d_out, output_dim)
        self.assertEqual(info["type"], "PCA")
        self.assertEqual(info["input_dimension"], input_dim)
        self.assertEqual(info["output_dimension"], output_dim)
        self.assertTrue(info["requires_training"])

        # Test L2 normalization
        transform, info = create_transformation("NORM", input_dim)

        self.assertIsInstance(transform, faiss.NormalizationTransform)
        self.assertEqual(transform.d, input_dim)
        self.assertEqual(info["type"], "NORM")
        self.assertEqual(info["input_dimension"], input_dim)
        self.assertEqual(info["output_dimension"], input_dim)
        self.assertFalse(info["requires_training"])

        # Test OPQ transformation
        transform, info = create_transformation("OPQ", input_dim, output_dim, M=8)

        self.assertIsInstance(transform, faiss.OPQMatrix)
        self.assertEqual(transform.d_in, input_dim)
        self.assertEqual(transform.d_out, output_dim)
        self.assertEqual(transform.M, 8)
        self.assertEqual(info["type"], "OPQ")
        self.assertEqual(info["input_dimension"], input_dim)
        self.assertEqual(info["output_dimension"], output_dim)
        self.assertTrue(info["requires_training"])
        self.assertEqual(info["M"], 8)

    def test_create_pretransform_index(self):
        """Test creating an IndexPreTransform with a base index."""
        # Create a transformation
        input_dim = 128
        output_dim = 64
        transform, transform_info = create_transformation("PCA", input_dim, output_dim)

        # Create a base index
        base_index = faiss.IndexFlatL2(output_dim)

        # Create the pretransform index
        index, info = create_pretransform_index(base_index, transform, transform_info)

        self.assertIsInstance(index, faiss.IndexPreTransform)
        self.assertEqual(index.d, input_dim)
        self.assertEqual(index.index.d, output_dim)
        self.assertEqual(info["type"], "IndexPreTransform")
        self.assertEqual(info["transform_type"], "PCA")
        self.assertEqual(info["input_dimension"], input_dim)
        self.assertEqual(info["output_dimension"], output_dim)
        self.assertEqual(info["base_index_type"], "IndexFlatL2")
        self.assertTrue(info["requires_training"])

    def test_transform_training(self):
        """Test training a transformation."""
        # Create a PCA transformation
        input_dim = 128
        output_dim = 64
        transform, _ = create_transformation("PCA", input_dim, output_dim)

        # Generate random training data
        np.random.seed(42)  # For reproducibility
        training_vectors = np.random.random((1000, input_dim)).astype(np.float32)

        # Check training status before training
        self.assertFalse(is_transform_trained(transform))

        # Train the transform
        success = train_transform(transform, training_vectors)

        self.assertTrue(success)
        self.assertTrue(is_transform_trained(transform))

        # Test applying the transform
        vector = np.random.random(input_dim).astype(np.float32)
        # Apply the transform directly without storing in an unused variable
        transform.apply_py(vector.reshape(1, -1))

        # Verify the output dimension
        transformed_output = transform.apply_py(vector.reshape(1, -1))
        self.assertEqual(transformed_output.shape, (1, output_dim))

    def test_get_transform_training_requirements(self):
        """Test getting training requirements for transformations."""
        # PCA transformation
        input_dim = 128
        output_dim = 64
        transform, _ = create_transformation("PCA", input_dim, output_dim)

        requirements = get_transform_training_requirements(transform)

        self.assertTrue(requirements["requires_training"])
        self.assertFalse(requirements["is_trained"])
        self.assertGreater(requirements["min_training_vectors"], 0)
        self.assertGreater(
            requirements["recommended_training_vectors"],
            requirements["min_training_vectors"]
        )

        # L2 normalization (doesn't require training)
        transform, _ = create_transformation("NORM", input_dim)

        requirements = get_transform_training_requirements(transform)

        self.assertFalse(requirements["requires_training"])
        self.assertTrue(requirements["is_trained"])
        self.assertEqual(requirements["min_training_vectors"], 0)


if __name__ == "__main__":
    unittest.main()
