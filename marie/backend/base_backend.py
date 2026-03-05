from abc import ABC, abstractmethod


class DocumentBackend(ABC):
    @classmethod
    @abstractmethod
    def supported_formats(cls) -> set[str]:
        """Canonical format names this backend handles."""

    @abstractmethod
    def convert(self, file_path: str, **kwargs) -> dict:
        """Convert document to pipeline-compatible output.

        Returns dict with 'mode' key:
          - mode='frames': {'mode': 'frames', 'frames': List[np.ndarray]}
          - mode='parsed': {'mode': 'parsed', 'results': List[Dict], 'pages': int}
        """

    @classmethod
    def is_available(cls) -> bool:
        return True
