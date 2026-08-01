import numpy

from Plotting.Metrics import Compute_Meltpool_Dimensions


def test_compute_meltpool_dimensions_handles_empty_and_flattened_data():
    node_coords = numpy.array(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        ]
    )
    data = numpy.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        ]
    )

    length, width, depth = Compute_Meltpool_Dimensions(
        data=data,
        node_coords=node_coords,
        threshold=0.5,
        n_for_avg=1,
    )

    numpy.testing.assert_allclose(length, [0.0, 2.0, 2.0])
    numpy.testing.assert_allclose(width, [0.0, 0.0, 1.0])
    numpy.testing.assert_allclose(depth, [0.0, 0.0, 1.0])


def test_compute_meltpool_dimensions_accepts_cnn_shape_and_transposed_coords():
    node_coords = numpy.array(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        ]
    )
    data_flat = numpy.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        ]
    )
    data_cnn = data_flat.reshape(3, 1, 2, 3)

    length, width, depth = Compute_Meltpool_Dimensions(
        data=data_cnn,
        node_coords=node_coords.T,
        threshold=0.5,
        n_for_avg=10,
    )

    # n_for_avg is larger than the number of melted nodes, so each available melted node is used.
    numpy.testing.assert_allclose(length, [0.0, 0.0, 0.0])
    numpy.testing.assert_allclose(width, [0.0, 0.0, 0.0])
    numpy.testing.assert_allclose(depth, [0.0, 0.0, 0.0])
