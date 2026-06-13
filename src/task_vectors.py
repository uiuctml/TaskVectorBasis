import abc

import torch


class _TaskVector(abc.ABC):
    def __init__(
        self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
            self._param_names = [k for k in self.vector.keys()]
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                # load full models so we can get named_parameters() order for masks
                pretrained_model = self._load_checkpoint(pretrained_checkpoint)
                finetuned_model  = self._load_checkpoint(finetuned_checkpoint)

                if not isinstance(pretrained_model, dict):
                    pretrained_sd = pretrained_model.state_dict()
                else:
                    pretrained_sd = pretrained_model

                if not isinstance(finetuned_model, dict):
                    finetuned_sd  = finetuned_model.state_dict()
                else:
                    finetuned_sd = finetuned_model

                # use finetuned model’s parameter order; filter out int64/uint8
                ordered_names = []
                named_params = finetuned_model.named_parameters() if not isinstance(finetuned_model, dict) else zip(finetuned_sd.keys(), finetuned_sd.values())
                for name, _ in named_params:
                    dt = finetuned_sd[name].dtype
                    if dt not in (torch.int64, torch.uint8):
                        ordered_names.append(name)

                self._param_names = ordered_names

                self.vector = {}
                for name in ordered_names:
                    self.vector[name] = finetuned_sd[name] - pretrained_sd[name]

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        raise NotImplementedError

    @abc.abstractmethod
    def _cast_to_same_type(self, other):
        raise NotImplementedError

    def __add__(self, other):
        """Add two task vectors together."""
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return self.__class__(vector=new_vector)

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other):
        """Dot product of two task vectors."""
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = self._load_checkpoint(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"  # noqa: E501
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return torch.load(checkpoint, map_location="cpu")

    def apply_to_nonlinear(self, pretrained_nonlinear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a nonlinear pretrained model."""
        return self.apply_to(pretrained_nonlinear_checkpoint, scaling_coef)

    def _cast_to_same_type(self, other):
        if isinstance(other, NonLinearTaskVector):
            return other
        return NonLinearTaskVector(vector=other.vector)
