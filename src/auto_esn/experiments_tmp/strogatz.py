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


    def watts_strogatz_graph(neighbours, rewire_proba, seed=1):
        input_weights = CompositeInitializer().with_seed(seed).uniform()
        hidden_weight = CompositeInitializer() \
            .with_seed(seed) \
            .uniform() \
            .watts_strogatz(neighbours=neighbours, rewire_proba=rewire_proba) \
            .spectral_normalize() \
            .scale(1.0)

        return WeightInitializer(
            weight_hh_init=hidden_weight,
            weight_ih_init=input_weights
        )


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
        initialization_fun=watts_strogatz_graph,
        initialization_parameter_space={
            "neighbours": OneOf([250, 100, 50, 25, 15, 10, 7, 5, 3]),
            "rewire_proba": OneOf([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 0.75, 0.85, 0.95])
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
        # PredefinedDataset(DatasetType.SUNSPOT).load(val_size=700, test_size=700),
        # PredefinedDataset(DatasetType.MultipleSuperimposedOscillators).load(val_size=1000, test_size=1000),
        # PredefinedDataset(DatasetType.MackeyGlass).load(val_size=200, test_size=200),
        PredefinedDataset(DatasetType.Temperature).load(val_size=300, test_size=300),
    ]

    experiment = Experiment(
        model_generator=model_generator,
        datasets=datasets,
        metric=NRMSELoss()
    )
    experiment.run("C:\Experiments\ESN with Graph init\Watts_strogatz", trials_per_configuration=6)
