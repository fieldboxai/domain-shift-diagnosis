from typing import List

import numpy as np
import pandas as pd

from .source_data import SourceDataSampler


class UnitaryShiftSampler:
    def __init__(self, source_data_sampler: SourceDataSampler) -> None:

        self.source_data_sampler = source_data_sampler
        self.shift_counter = 0
        self.std_z = pd.Series(
            np.ones(len(source_data_sampler.z_columns)),
            index=source_data_sampler.z_columns,
        )
        self.std_x = pd.Series(
            np.sqrt(np.sum(np.square(self.source_data_sampler.C.values), 0)),
            index=source_data_sampler.x_columns,
        )

    def linear_shift(self, x: np.array, a: float, b: float) -> np.array:
        return a * x + b

    def sample(
        self,
        source_gaussian_noise_std: float,
        shift_source_list: List[str],
        n_samples_list: List[int],
        shift_intensity_list: List[int],
        gaussian_noise_std_list: List[int],
        random_state: int,
    ) -> pd.DataFrame:

        Z_target = pd.DataFrame()
        X_target = pd.DataFrame()
        noise = pd.DataFrame()

        for n_samples in n_samples_list:
            Z_source, X_source, _ = self.source_data_sampler.sample(
                n_samples, source_gaussian_noise_std, random_state
            )
            self.n_samples = n_samples

            for gaussian_noise_std in gaussian_noise_std_list:

                self.gaussian_noise_std = gaussian_noise_std

                for shift_intensity in shift_intensity_list:

                    for shift_source in shift_source_list:

                        (
                            Z_concept_shift,
                            X_concept_shift,
                        ) = self.generate_concept_shift(
                            X_source, shift_intensity, shift_source
                        )
                        (
                            Z_covariate_shift,
                            X_covariate_shift,
                        ) = self.generate_covariate_shift(
                            Z_source, shift_intensity, shift_source
                        )

                        _Z_target = pd.concat(
                            [Z_concept_shift, Z_covariate_shift],
                            axis=0,
                            ignore_index=False,
                        )
                        _X_target = pd.concat(
                            [X_concept_shift, X_covariate_shift],
                            axis=0,
                            ignore_index=False,
                        )
                        _noise = pd.DataFrame(
                            self.source_data_sampler.gaussian_noise(
                                _X_target.shape[0], gaussian_noise_std
                            ),
                            index=_X_target.index,
                            columns=self.source_data_sampler.x_columns,
                        )

                        Z_target = Z_target.append(_Z_target, ignore_index=False)
                        X_target = X_target.append(_X_target)
                        noise = noise.append(_noise)

                        print(f"Shift ID: {self.shift_counter}")

        X_target_noisy = X_target + noise.values

        return Z_target, X_target, X_target_noisy

    def generate_concept_shift(
        self,
        X_source: pd.DataFrame,
        shift_intensity: float,
        shift_source: str,
    ) -> pd.DataFrame:
        """
        Applies unitary shift to each column of X_source.
        """

        shift_intensity_weights = self.std_x
        X_target = self._generate_unitary_shifts(
            X_source, "Concept", shift_source, shift_intensity, shift_intensity_weights
        )
        Z_target = pd.DataFrame(
            data=X_target.values @ self.source_data_sampler.C_inv.values,
            index=X_target.index,
            columns=self.source_data_sampler.z_columns,
        )
        return Z_target, X_target

    def generate_covariate_shift(
        self,
        Z_source: pd.DataFrame,
        shift_intensity: float,
        shift_source: str,
    ) -> pd.DataFrame:
        """
        Applies unitary shift to each column of Z_source.
        """
        shift_intensity_weights = self.std_z
        Z_target = self._generate_unitary_shifts(
            Z_source,
            "Covariate",
            shift_source,
            shift_intensity,
            shift_intensity_weights,
        )
        X_target = Z_target.values @ self.source_data_sampler.C.values

        X_target = pd.DataFrame(
            X_target,
            index=Z_target.index,
            columns=self.source_data_sampler.x_columns,
        )
        return Z_target, X_target

    def _generate_unitary_shifts(
        self,
        X_source: pd.DataFrame,
        shift_type: str,
        shift_source: str,
        shift_intensity: float,
        shift_intensity_weights: pd.Series,
    ) -> pd.DataFrame:
        """Iteratively applies a linear transformation to each column
        of X_source. All the shifted datasets are stacked in a single dataframe.
        """

        output = pd.DataFrame()

        for x_col in X_source.columns:

            is_a_shifted = shift_source == "Scale"
            is_b_shifted = shift_source == "Loc"

            X_target = X_source.copy(deep=True)

            shift = shift_intensity * shift_intensity_weights[x_col]

            X_target[x_col] = self.linear_shift(
                x=X_source[x_col].values,
                a=is_a_shifted * shift,
                b=is_b_shifted * shift,
            )

            X_target["Shift Type"] = shift_type
            X_target["Shift Source"] = shift_source
            X_target["Shift Intensity"] = shift_intensity
            X_target["shift_id"] = self.shift_counter
            X_target["Gaussian Noise Std"] = self.gaussian_noise_std
            X_target["N Samples"] = self.n_samples

            output = output.append(X_target, ignore_index=True)
            self.shift_counter += 1

        return output.set_index(
            [
                "Shift Type",
                "Shift Source",
                "Shift Intensity",
                "Gaussian Noise Std",
                "N Samples",
                "shift_id",
            ],
            drop=True,
        )
