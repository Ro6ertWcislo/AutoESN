import logging

from auto_esn.datasets.predefined import PredefinedDataset, DatasetType
from auto_esn.esn.esn import GroupedDeepESN
from auto_esn.esn.reservoir.activation import self_normalizing_default
from auto_esn.esn.reservoir.initialization import WeightInitializer, CompositeInitializer
from auto_esn.esn.reservoir.util import NRMSELoss
from auto_esn.experiments.Experiment import Experiment, default_seed_generator, GridSearchGeneratorProvider, OneOf, \
    Just, ESNModelGenerator

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)


    def regular_graph(degree, seed=1):
        i2 = CompositeInitializer().with_seed(seed).uniform()
        i = CompositeInitializer() \
            .with_seed(seed) \
            .uniform() \
            .regular_graph(degree) \
            .spectral_normalize() \
            .scale(1.0)

        w = WeightInitializer()
        w.weight_hh_init = i
        w.weight_ih_init = i2
        return w


    model_generator = ESNModelGenerator(
        model_class=GroupedDeepESN,
        model_parameter_space={
            "input_size": Just(1),
            "bias": Just(False),
            "num_layers": Just([1]),
            "groups": Just(1),
            "washout": Just(100),
            "hidden_size": Just(1000),

        },
        initialization_fun=regular_graph,
        initialization_parameter_space={
            "degree": OneOf([250, 100, 50, 25, 15, 10, 7, 5, 3])
        },
        activation_fun=self_normalizing_default,
        activation_parameter_space={
            "leaky_rate": Just(1.0),
            "spectral_radius": OneOf([2500, 1000, 250, 100, 50, 15])
        },
        generator_provider=GridSearchGeneratorProvider(),
        seed_generator=default_seed_generator(),
    )

    datasets = [
        PredefinedDataset(DatasetType.SUNSPOT).load(val_size=700, test_size=700),
        PredefinedDataset(DatasetType.MultipleSuperimposedOscillators).load(val_size=1000, test_size=1000),
        PredefinedDataset(DatasetType.MackeyGlass).load(val_size=200, test_size=200),
    ]

    experiment = Experiment(
        model_generator=model_generator,
        datasets=datasets,
        metric=NRMSELoss()
    )
    experiment.run("C:\Experiments\ESN with Graph init\Regular_graph", trials_per_configuration=6)
