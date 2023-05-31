import collections

import deepdanbooru as dd
import numpy as np
import tensorflow as tf


class deepdanbooru_pipeline:
    def __init__(self, projectpath):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=4096
                        )
                    ],
                )
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

        self.classif_model = dd.project.load_model_from_project(projectpath)
        self.all_tags = dd.project.load_tags_from_project(projectpath)
        self.model_width = self.classif_model.input_shape[2]
        self.model_height = self.classif_model.input_shape[1]
        self.all_tags = np.array(self.all_tags)

    def __call__(self, imagepath, threshold, multiplier, prefix):
        image = dd.data.load_image_for_evaluate(
            imagepath, width=self.model_width, height=self.model_height
        )
        image = np.array([image])

        result = self.classif_model.predict(image).reshape(-1, self.all_tags.shape[0])[
            0
        ]

        result_tags = {}
        for i in range(len(self.all_tags)):
            if result[i] > threshold:
                result_tags[self.all_tags[i]] = result[i]
        sorted_tags = reversed(sorted(result_tags.keys(), key=lambda x: result_tags[x]))
        sorted_results = collections.OrderedDict()
        for tag in sorted_tags:
            sorted_results[tag] = result_tags[tag]

        prompt = prefix
        for tag, prob in sorted_results.items():
            tag_strength = prob * multiplier
            prompt += f"({tag}){tag_strength}, "
        prompt = prompt[:-1]

        return prompt
