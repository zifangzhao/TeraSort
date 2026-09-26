import unittest

import numpy as np

from terasort.session_template_merge import build_template_map


class TemplateMergeTests(unittest.TestCase):
    def test_merges_only_close_same_shank_waveform_duplicates(self):
        base = np.zeros(61, np.float32)
        base[27:34] = [-.2, -.5, -1., -.6, -.3, -.1, -.05]
        waveforms = np.zeros((4, 61, 2), np.float32)
        waveforms[0] = np.column_stack((base, .4 * base))
        waveforms[1] = np.column_stack((.4 * base, base))
        waveforms[2] = np.column_stack((base, .4 * base))
        waveforms[3] = np.column_stack((.4 * base, base))
        channels = np.asarray([[0, 1], [1, 0], [2, 3], [3, 2]], np.int32)
        anchors = np.asarray([0, 1, 2, 3], np.int32)
        geometry = np.asarray([[0, 0], [32, 0], [64, 0], [96, 0]], np.float32)
        shanks = np.asarray([0, 0, 1, 1], np.int32)

        mapping, stats = build_template_map(
            waveforms, channels, anchors, geometry, shanks,
            cosine_threshold=.93, radius_um=32.,
        )

        self.assertEqual(mapping[0], mapping[1])
        self.assertEqual(mapping[2], mapping[3])
        self.assertNotEqual(mapping[0], mapping[2])
        self.assertEqual(stats["output_groups"], 2)
        self.assertEqual(stats["links_accepted"], 2)

    def test_rejects_invalid_linking_parameters(self):
        with self.assertRaisesRegex(ValueError, "parameters"):
            build_template_map(
                np.ones((1, 61, 1), np.float32),
                np.asarray([[0]], np.int32), np.asarray([0], np.int32),
                np.asarray([[0., 0.]]), np.asarray([0]), radius_um=0,
            )


if __name__ == "__main__":
    unittest.main()
