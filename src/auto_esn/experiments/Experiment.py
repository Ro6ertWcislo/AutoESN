import copy
import csv
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Callable, Any, List, Tuple, Generator

import numpy as np

from auto_esn.Experiment import Dataset
from auto_esn.esn.esn import ESNBase
from auto_esn.esn.reservoir.activation import Activation
from auto_esn.esn.reservoir.initialization import WeightInitializer

SeedGenerator = Callable[[], int]


def default_seed_generator():
    i = 0
    while True:
        i += 7
        yield i


class Parameter:
    pass


@dataclass
class Just(Parameter):
    value: Any


@dataclass
class OneOf(Parameter):
    value: List[Any]


class GeneratorProvider:

    def create_generator(self, parameter_space: Dict[str, Parameter]) -> Dict[str, Any]:
        pass


def backtrack(actual_state: Dict[str, Any], parameters_left: List[Tuple[str, List[Any]]], current_position):
    if len(parameters_left) == current_position:
        yield copy.deepcopy(actual_state)
    else:
        current_params = parameters_left[current_position]
        param_name, params_values = current_params
        for value in params_values:
            actual_state[param_name] = value
            yield from backtrack(actual_state, parameters_left, current_position + 1)


class GridSearchGeneratorProvider(GeneratorProvider):

    def create_generator(self, parameter_space: Dict[str, Parameter]):
        single_values = {}
        actual_parameter_space = []
        for parameter_name, parameters in parameter_space.items():
            if isinstance(parameters, Just):
                single_values[parameter_name] = parameters.value
            elif isinstance(parameters, OneOf):
                actual_parameter_space.append((parameter_name, parameters.value))

        return backtrack(single_values, actual_parameter_space, 0)


class UniformRandomGeneratorProvider(GeneratorProvider):  # todo dummy version so far

    def create_generator(self, parameter_space: Dict[str, Parameter]) -> Dict[str, Any]:
        pass


class ModelGenerator:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class ModelInitializer:
    """
    Abstraction over a set of parameters and model.
    Enables to instantiate the same model "recipe" with different initial conditions either randomply or with specified seeds
    """

    def __init__(self,
                 model_class,
                 model_parameters,
                 activation_fun,
                 activation_parameters,
                 initialization_fun,
                 initialization_parameters,
                 seed_generator):
        self.model_class = model_class
        self.model_parameters = model_parameters
        self.activation_fun = activation_fun
        self.activation_parameters = activation_parameters
        self.initialization_fun = initialization_fun
        self.initialization_parameters = initialization_parameters
        self.seed_generator = seed_generator

    def initialize(self) -> ESNBase:
        try:
            initializer = self.initialization_fun(seed=next(self.seed_generator), **self.initialization_parameters)
        except TypeError:
            # no seed in initializer. Dirty but works :) todo make it prettier with wrapping initialization in class
            initializer = self.initialization_fun(**self.initialization_parameters)
        return self.model_class(
            initializer=initializer,
            activation=self.activation_fun(**self.activation_parameters),
            **self.model_parameters
        )


class ESNModelGenerator(ModelGenerator):

    def __init__(self,
                 model_class: ESNBase.__class__,
                 model_parameter_space: Dict[str, Parameter],
                 initialization_fun: Callable[[Any], WeightInitializer],
                 initialization_parameter_space: Dict[str, Parameter],
                 activation_fun: Callable[[Any], Activation],
                 activation_parameter_space: Dict[str, Parameter],
                 generator_provider: GeneratorProvider,
                 seed_generator: Generator[int, Any, None] = default_seed_generator):
        self.model_class = model_class
        self.model_parameter_space = model_parameter_space
        self.initialization_fun = initialization_fun
        self.initialization_parameter_space = initialization_parameter_space
        self.activation_fun = activation_fun
        self.activation_parameter_space = activation_parameter_space
        self.sample_generator = generator_provider.create_generator(
            {**model_parameter_space,
             **initialization_parameter_space,
             **activation_parameter_space}
        )
        self.seed_generator = seed_generator

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[ModelInitializer, Dict[str, Any]]:
        parameters = next(self.sample_generator)
        model_parameters = self._filter_dict_by_keys(parameters, self.model_parameter_space)
        initialization_parameters = self._filter_dict_by_keys(parameters, self.initialization_parameter_space)
        activation_parameters = self._filter_dict_by_keys(parameters, self.activation_parameter_space)

        return ModelInitializer(
            model_class=self.model_class,
            model_parameters=model_parameters,
            initialization_fun=self.initialization_fun,
            initialization_parameters=initialization_parameters,
            activation_fun=self.activation_fun,
            activation_parameters=activation_parameters,
            seed_generator=self.seed_generator
        ), parameters

    def _filter_dict_by_keys(self, origin: Dict[Any, Any], mask: [Any, Any]):
        return dict(filter(lambda key_val: key_val[0] in mask, origin.items()))


class Experiment:
    def __init__(self,
                 model_generator: ModelGenerator,
                 datasets: List[Dataset],
                 metric):
        self.model_generator = model_generator
        self.datasets = datasets
        self.metric = metric

    def run(self, destination_folder: str, trials_per_configuration=1, iteration_limit: int = None):
        iteration_count = 0
        iteration_limit = iteration_limit if iteration_limit is not None else float("inf")

        for model_initializer, parameters in self.model_generator:
            val_scores = defaultdict(list)
            test_scores = defaultdict(list)
            durations = defaultdict(list)
            for trial_no in range(trials_per_configuration):
                model = model_initializer.initialize()
                for dataset in self.datasets:
                    time_start = time.time()
                    model.fit(dataset.x_train, dataset.y_train)

                    output_val = model(dataset.x_val)
                    output_test = model(dataset.x_test) if dataset.x_test is not None else None

                    y_val = dataset.y_val
                    y_test = dataset.y_test
                    if dataset.spread is not None:
                        output_val = output_val * dataset.spread
                        y_val = y_val * dataset.spread
                        if output_test is not None:
                            output_test = output_test * dataset.spread
                            y_test = y_test * dataset.spread
                    if dataset.baseline is not None:
                        output_val = output_val + dataset.baseline
                        y_val = y_val + dataset.baseline
                        if output_test is not None:
                            output_test = output_test + dataset.baseline
                            y_test = y_test + dataset.baseline
                    val_score = self.metric(output_val, y_val).item()
                    val_scores[dataset.name].append(val_score)

                    if output_test is not None:
                        test_score = self.metric(output_test, y_test).item()
                        test_scores[dataset.name].append(test_score)
                    else:
                        test_score = None
                    time_end = time.time()
                    duration = (time_end - time_start)
                    durations[dataset.name].append(duration)

                    logging.info(
                        f"Training & Evaluation time: {duration}. Val score: {val_score}. Test score: {test_score}. Trained model no: {iteration_count}. Trial no: {trial_no} for current configuration: {parameters} for dataset {dataset.name}")
                    iteration_count += 1
                    if iteration_count > iteration_limit:
                        continue
            for dataset in self.datasets:
                self.save_results(val_scores, test_scores, durations, parameters, dataset.name, destination_folder)
            if iteration_count > iteration_limit:
                return

    def get_row_and_columns(self, val_results: Dict[str, List], test_results: Dict[str, List], parameters: Dict,
                            durations: Dict[str, List], name: str):
        def get_stats(l: List):
            if len(l) == 0:
                return None, None, None, None
            return np.average(l), np.std(l), np.min(l), np.max(l)

        val_avg, val_std, val_min, val_max = get_stats(val_results[name])
        test_avg, test_std, test_min, test_max = get_stats(test_results[name])

        all_dict = {
            "val_avg": val_avg,
            "val_std": val_std,
            "val_min": val_min,
            "val_max": val_max,
            "test_avg": test_avg,
            "test_std": test_std,
            "test_min": test_min,
            "test_max": test_max,
            "avg_duration": np.average(durations[name]),
            **parameters
        }
        column_value_pairs = [x for x in sorted(all_dict.items(), key=lambda x: x[0])]
        headers, values = zip(*column_value_pairs)
        return headers, values

    def save_results(self, val_scores, test_scores, durations, parameters, name, destination_folder):
        headers, values = self.get_row_and_columns(val_scores, test_scores, parameters, durations, name)
        destination_path = f"{destination_folder}/{name}.csv"
        if not os.path.exists(destination_path):
            with open(destination_path, mode='w') as csv_result:
                csv_writer = csv.writer(csv_result, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(headers)
                csv_writer.writerow(values)
        else:
            with open(destination_path, mode='a+') as csv_result:
                csv_writer = csv.writer(csv_result, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(values)
