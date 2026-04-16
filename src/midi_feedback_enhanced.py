"""
Enhanced MIDI Feedback System
Uses improved generator and provides better optimization
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import pickle
import warnings

# Import enhanced modules
from midi_generate_enhanced import EnhancedMIDIGenerator, enhanced_process_audio
from midi_evaluator import ScoreEvaluator


def _convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: _convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_python_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


@dataclass
class EnhancedParameterSet:
    """Complete parameter set for enhanced MIDI generator"""
    # Audio processing
    sr: int = 44100
    hop_length: int = 512
    frame_length: int = 2048

    # Frequency range
    min_freq: int = 80
    max_freq: int = 1200

    # Pitch detection
    pitch_method: str = 'pyin'  # 'pyin', 'crepe', 'hybrid'
    n_thresholds: int = 200
    resolution: float = 0.1

    # Voicing detection
    voicing_threshold: float = 0.6
    confidence_weight: float = 0.5
    use_harmonic_enhancement: bool = True
    harmonic_margin: int = 3

    # Post-processing
    median_filter_size: int = 5
    smoothing_window: int = 3
    interpolate_missing: bool = True

    # Note segmentation
    onset_threshold: float = 0.5
    min_note_duration: float = 0.1
    max_gap: float = 0.05
    min_velocity: int = 40
    max_velocity: int = 127

    # Polyphony (experimental)
    max_polyphony: int = 1
    polyphony_threshold: float = 0.3

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedParameterSet':
        return cls(**data)

    def to_generator_config(self) -> Dict[str, Any]:
        """Convert to configuration dictionary for EnhancedMIDIGenerator"""
        return self.to_dict()


class DatasetCollector:
    """
    Collects training data from audio-MIDI pairs and their evaluations
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)

        # Dataset storage
        self.dataset_file = storage_path / "training_dataset.jsonl"
        self.metadata_file = storage_path / "dataset_metadata.json"

        # Initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load dataset metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'n_samples': 0,
            'audio_files': [],
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat()
        }

    def _save_metadata(self):
        """Save dataset metadata"""
        self.metadata['updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(_convert_to_python_types(self.metadata), f, indent=2)

    def add_sample(self, audio_path: Path, reference_midi_path: Path,
                   params: EnhancedParameterSet, evaluation: Dict[str, float]):
        """
        Add a training sample to the dataset

        Args:
            audio_path: Path to audio file
            reference_midi_path: Path to reference MIDI file
            params: Parameters used for generation
            evaluation: Evaluation metrics
        """
        sample = {
            'audio_file': str(audio_path.relative_to(self.storage_path) if audio_path.is_relative_to(self.storage_path) else audio_path),
            'reference_midi': str(reference_midi_path.relative_to(self.storage_path) if reference_midi_path.is_relative_to(self.storage_path) else reference_midi_path),
            'params': params.to_dict(),
            'evaluation': evaluation,
            'timestamp': datetime.now().isoformat()
        }

        # Save to JSONL file
        with open(self.dataset_file, 'a') as f:
            f.write(json.dumps(_convert_to_python_types(sample)) + '\n')

        # Update metadata
        self.metadata['n_samples'] += 1
        if str(audio_path) not in self.metadata['audio_files']:
            self.metadata['audio_files'].append(str(audio_path))

        self._save_metadata()
        print(f"Added sample {self.metadata['n_samples']} to dataset")

    def get_dataset(self) -> List[Dict]:
        """Load entire dataset"""
        dataset = []
        if self.dataset_file.exists():
            with open(self.dataset_file, 'r') as f:
                for line in f:
                    if line.strip():
                        dataset.append(json.loads(line))
        return dataset

    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset statistics and patterns"""
        dataset = self.get_dataset()
        if not dataset:
            return {"error": "Dataset is empty"}

        # Basic statistics
        n_samples = len(dataset)
        scores = [sample['evaluation']['composite_score'] for sample in dataset]
        f1_scores = [sample['evaluation']['f1'] for sample in dataset]

        # Parameter distributions
        param_stats = {}
        if n_samples > 0:
            sample_params = dataset[0]['params'].keys()
            for param in sample_params:
                values = [sample['params'][param] for sample in dataset]
                if isinstance(values[0], (int, float)):
                    param_stats[param] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

        # Correlation analysis
        correlations = {}
        if n_samples > 5:
            for param in list(param_stats.keys())[:10]:  # Limit to first 10 params
                param_values = [sample['params'][param] for sample in dataset]
                if isinstance(param_values[0], (int, float)):
                    try:
                        corr = np.corrcoef(param_values, scores)[0, 1]
                        if not np.isnan(corr):
                            correlations[param] = float(corr)
                    except:
                        pass

        return {
            'n_samples': n_samples,
            'score_stats': {
                'mean': np.mean(scores) if scores else 0,
                'std': np.std(scores) if scores else 0,
                'min': np.min(scores) if scores else 0,
                'max': np.max(scores) if scores else 0
            },
            'f1_stats': {
                'mean': np.mean(f1_scores) if f1_scores else 0,
                'std': np.std(f1_scores) if f1_scores else 0,
                'min': np.min(f1_scores) if f1_scores else 0,
                'max': np.max(f1_scores) if f1_scores else 0
            },
            'param_stats': param_stats,
            'correlations': correlations,
            'suggestions': self._generate_suggestions(correlations)
        }

    def _generate_suggestions(self, correlations: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on correlations"""
        suggestions = []
        for param, corr in correlations.items():
            if abs(corr) > 0.3:  # Significant correlation
                if corr > 0:
                    suggestions.append(f"Increase {param} (correlation with score: {corr:.3f})")
                else:
                    suggestions.append(f"Decrease {param} (correlation with score: {corr:.3f})")
        return suggestions[:5]  # Top 5 suggestions


class EnhancedFeedbackSystem:
    """
    Enhanced feedback system with improved optimization and dataset collection
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("enhanced_feedback_data")
        self.storage_path.mkdir(exist_ok=True)

        # Components
        self.dataset_collector = DatasetCollector(self.storage_path / "dataset")
        self.optimization_history = self.storage_path / "optimization"
        self.optimization_history.mkdir(exist_ok=True)

        # State
        self.best_params: Optional[EnhancedParameterSet] = None
        self.best_score: float = -float('inf')
        self.generator_cache: Dict[str, EnhancedMIDIGenerator] = {}

    def optimize_for_audio(self, audio_bytes: bytes, reference_midi_path: Path,
                          n_iterations: int = 30,
                          strategy: str = "bayesian") -> EnhancedParameterSet:
        """
        Optimize parameters for specific audio using advanced strategies

        Args:
            audio_bytes: Audio file bytes
            reference_midi_path: Path to reference MIDI
            n_iterations: Number of optimization iterations
            strategy: 'random', 'grid', 'bayesian', or 'adaptive'

        Returns:
            Optimized parameter set
        """
        print(f"Starting optimization for audio ({strategy} strategy, {n_iterations} iterations)")

        # Create temporary directory
        temp_dir = self.storage_path / "temp"
        temp_dir.mkdir(exist_ok=True)

        best_params = None
        best_score = -float('inf')
        history = []

        for i in range(n_iterations):
            print(f"\nIteration {i+1}/{n_iterations}")

            # Generate parameters based on strategy
            if strategy == "adaptive" and i > 0:
                params = self._adaptive_sample(history)
            elif strategy == "grid":
                params = self._grid_sample(i, n_iterations)
            else:  # random or bayesian (simplified)
                params = self._random_sample()

            # Generate and evaluate MIDI
            try:
                score, metrics = self._evaluate_with_params(
                    audio_bytes, reference_midi_path, params, temp_dir, i
                )

                # Record
                record = {
                    'iteration': i,
                    'params': params.to_dict(),
                    'metrics': metrics,
                    'score': score,
                    'strategy': strategy
                }
                history.append(record)

                print(f"  Score: {score:.4f} (F1: {metrics['f1']:.3f}, "
                      f"Precision: {metrics['precision']:.3f})")

                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"  New best!")

                    # Also add to dataset
                    self.dataset_collector.add_sample(
                        audio_path=temp_dir / f"audio_{i}.wav",
                        reference_midi_path=reference_midi_path,
                        params=params,
                        evaluation=metrics
                    )

            except Exception as e:
                print(f"  Iteration failed: {e}")
                continue

        # Save optimization results
        if best_params:
            self._save_optimization(best_params, best_score, history, strategy)

        # Cleanup
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
            except:
                pass

        print(f"\nOptimization completed. Best score: {best_score:.4f}")
        return best_params or EnhancedParameterSet()

    def _evaluate_with_params(self, audio_bytes: bytes, reference_midi_path: Path,
                             params: EnhancedParameterSet, temp_dir: Path,
                             iteration: int) -> Tuple[float, Dict]:
        """Generate MIDI with given parameters and evaluate"""
        # Create generator with parameters
        config = params.to_generator_config()
        generator = EnhancedMIDIGenerator(config)

        # Generate MIDI
        midi = generator.process_audio(audio_bytes)

        # Save temporary files
        audio_temp = temp_dir / f"audio_{iteration}.wav"
        midi_temp = temp_dir / f"generated_{iteration}.mid"

        with open(audio_temp, 'wb') as f:
            f.write(audio_bytes)

        midi.write(str(midi_temp))

        # Evaluate
        evaluator = ScoreEvaluator(reference_midi_path, midi_temp)
        accuracy = evaluator.note_accuracy()
        rhythm = evaluator.rhythm_analysis()

        # Composite score (customizable weights)
        weights = {
            'f1': 0.4,
            'precision': 0.2,
            'recall': 0.1,
            'timing': 0.2,
            'rhythm': 0.1
        }

        score = (
            weights['f1'] * accuracy['f1'] +
            weights['precision'] * accuracy['precision'] +
            weights['recall'] * accuracy['recall'] +
            weights['timing'] * (1 - min(rhythm['avg_time_deviation'] / 5, 1)) +
            weights['rhythm'] * (1 - min(rhythm['dtw_distance'] / 500, 1))
        )

        metrics = {
            'f1': accuracy['f1'],
            'precision': accuracy['precision'],
            'recall': accuracy['recall'],
            'dtw_distance': rhythm['dtw_distance'],
            'avg_time_deviation': rhythm['avg_time_deviation'],
            'max_deviation': rhythm['max_deviation'],
            'composite_score': score
        }

        return score, metrics

    def _random_sample(self) -> EnhancedParameterSet:
        """Sample random parameters with improved ranges based on common audio characteristics"""
        # Improved parameter ranges for better accuracy
        # Based on analysis: audio typically has pitch range 80-2000Hz for most instruments
        # hop_length 256-512 provides good time-frequency tradeoff
        # Note durations typically 0.08-0.5 seconds for musical notes
        return EnhancedParameterSet(
            sr=44100,  # Fixed to common sample rate
            hop_length=int(np.random.uniform(256, 512)),  # Better time resolution
            min_freq=int(np.random.uniform(80, 250)),  # Higher minimum to exclude noise
            max_freq=int(np.random.uniform(800, 2000)),  # Lower maximum for most instruments
            pitch_method=np.random.choice(['pyin', 'hybrid']),
            voicing_threshold=np.random.uniform(0.5, 0.8),  # Tighter range for better voicing detection
            confidence_weight=np.random.uniform(0.3, 0.7),  # Balanced confidence weighting
            min_note_duration=np.random.uniform(0.1, 0.25),  # Allow detection of short notes (47% of ref notes <0.3s)
            max_gap=np.random.uniform(0.05, 0.15),  # Allow natural gaps between notes
            onset_threshold=np.random.uniform(0.4, 0.7)  # Moderate onset sensitivity
        )

    def _grid_sample(self, iteration: int, total: int) -> EnhancedParameterSet:
        """Sample parameters in a grid-like pattern"""
        # Simplified grid sampling
        params = self._random_sample()
        return params

    def _adaptive_sample(self, history: List[Dict]) -> EnhancedParameterSet:
        """Adaptively sample parameters using weighted average based on scores"""
        if len(history) < 3:
            return self._random_sample()

        # Get top 5 records weighted by their scores
        sorted_records = sorted(history, key=lambda x: x['score'], reverse=True)[:5]
        scores = np.array([r['score'] for r in sorted_records])
        # Convert scores to weights (softmax-like, ensure positive)
        weights = np.exp(scores - np.max(scores))  # Numerical stability
        weights = weights / weights.sum()

        best_params = [EnhancedParameterSet.from_dict(r['params']) for r in sorted_records]

        # Weighted average with adaptive exploration noise
        optimized_params = {}
        exploration_factor = max(0.1, 0.3 - len(history) * 0.01)  # Decrease noise over time

        for field in EnhancedParameterSet.__dataclass_fields__:
            values = [getattr(p, field) for p in best_params]

            if isinstance(values[0], (int, float)):
                # Weighted average for numerical parameters
                weighted_avg = np.average(values, weights=weights)

                # Adaptive exploration noise
                noise_scale = exploration_factor * np.std(values) if len(values) > 1 else exploration_factor * abs(weighted_avg)

                if isinstance(values[0], int):
                    noisy_value = int(np.random.normal(weighted_avg, max(1, noise_scale)))
                    # Ensure within reasonable bounds
                    if field == 'min_freq':
                        noisy_value = max(50, min(noisy_value, 500))
                    elif field == 'max_freq':
                        noisy_value = max(500, min(noisy_value, 4000))
                    elif field == 'hop_length':
                        noisy_value = max(128, min(noisy_value, 1024))
                    optimized_params[field] = noisy_value
                else:
                    noisy_value = np.random.normal(weighted_avg, noise_scale)
                    # Clip to reasonable ranges
                    if field == 'voicing_threshold':
                        noisy_value = np.clip(noisy_value, 0.3, 0.9)
                    elif field == 'min_note_duration':
                        noisy_value = np.clip(noisy_value, 0.05, 0.5)
                    optimized_params[field] = float(noisy_value)
            else:
                # Weighted voting for categorical parameters
                unique_values = list(set(values))
                value_weights = []
                for val in unique_values:
                    val_weight = sum(weights[i] for i, v in enumerate(values) if v == val)
                    value_weights.append(val_weight)
                # Select with probability proportional to weight
                chosen_value = np.random.choice(unique_values, p=np.array(value_weights)/sum(value_weights))
                optimized_params[field] = chosen_value

        return EnhancedParameterSet(**optimized_params)

    def _save_optimization(self, params: EnhancedParameterSet, score: float,
                          history: List[Dict], strategy: str):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.optimization_history / f"optimization_{timestamp}.json"

        result = {
            'timestamp': timestamp,
            'strategy': strategy,
            'best_score': score,
            'best_params': params.to_dict(),
            'history': history,
            'analysis': self._analyze_optimization(history)
        }

        with open(result_file, 'w') as f:
            json.dump(_convert_to_python_types(result), f, indent=2)

        print(f"Optimization results saved to {result_file}")

    def _analyze_optimization(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze optimization results"""
        if not history:
            return {}

        scores = [r['score'] for r in history]
        improvements = []
        for i in range(1, len(scores)):
            improvements.append(scores[i] - scores[i-1])

        return {
            'final_score': scores[-1],
            'best_score': max(scores),
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_improvement': np.mean(improvements) if improvements else 0,
            'converged': len(scores) > 10 and np.std(scores[-5:]) < 0.01
        }

    def create_optimized_generator(self) -> EnhancedMIDIGenerator:
        """Create generator with optimized parameters"""
        if self.best_params:
            config = self.best_params.to_generator_config()
        else:
            config = EnhancedParameterSet().to_generator_config()

        return EnhancedMIDIGenerator(config)

    def get_recommendations(self) -> Dict[str, Any]:
        """Get recommendations based on collected data"""
        dataset_analysis = self.dataset_collector.analyze_dataset()

        recommendations = {
            'dataset_stats': dataset_analysis.get('score_stats', {}),
            'parameter_suggestions': dataset_analysis.get('suggestions', []),
            'best_practices': self._generate_best_practices(dataset_analysis)
        }

        return recommendations

    def _generate_best_practices(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate best practice recommendations"""
        practices = []

        param_stats = analysis.get('param_stats', {})
        if 'hop_length' in param_stats:
            stats = param_stats['hop_length']
            if stats['mean'] < 256:
                practices.append("Consider increasing hop_length for better frequency resolution")
            elif stats['mean'] > 768:
                practices.append("Consider decreasing hop_length for better temporal resolution")

        if 'min_note_duration' in param_stats:
            stats = param_stats['min_note_duration']
            if stats['mean'] < 0.08:
                practices.append("Minimum note duration may be too short, increasing could reduce false positives")

        if 'voicing_threshold' in param_stats:
            stats = param_stats['voicing_threshold']
            if stats['mean'] < 0.5:
                practices.append("Consider increasing voicing_threshold to reduce false pitch detections")

        return practices


class AudioInformedFeedbackSystem(EnhancedFeedbackSystem):
    """Feedback system with audio-informed parameter sampling"""

    def __init__(self, audio_bytes: bytes, storage_path: Path = None):
        super().__init__(storage_path)
        self.audio_bytes = audio_bytes
        self.audio_info = self._analyze_audio()

    def _analyze_audio(self):
        """Analyze audio to inform parameter ranges"""
        import tempfile
        import librosa
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(self.audio_bytes)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None)

            # Basic analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
            )
            voiced_f0 = f0[voiced_flag]

            info = {
                'sr': sr,
                'duration': len(y) / sr,
                'max_amplitude': np.max(np.abs(y)),
                'mean_amplitude': np.mean(np.abs(y))
            }

            if len(voiced_f0) > 0:
                info['min_freq'] = max(20, np.min(voiced_f0) * 0.8)
                info['max_freq'] = min(4000, np.max(voiced_f0) * 1.2)
                info['pitch_range_hz'] = (np.min(voiced_f0), np.max(voiced_f0))
            else:
                info['min_freq'] = 80
                info['max_freq'] = 2000
                info['pitch_range_hz'] = (None, None)

            # Spectral centroid for frequency content
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            info['spectral_centroid'] = np.mean(spectral_centroid)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return info

    def _random_sample(self) -> EnhancedParameterSet:
        """Sample random parameters informed by audio analysis"""
        sr = self.audio_info['sr']

        # Base hop_length on spectral centroid
        if self.audio_info['spectral_centroid'] < 500:
            # Lower frequencies, can use longer hop_length
            hop_min, hop_max = 256, 1024
        else:
            # Higher frequencies, need shorter hop_length
            hop_min, hop_max = 128, 512

        return EnhancedParameterSet(
            sr=sr,
            hop_length=int(np.random.uniform(hop_min, hop_max)),
            min_freq=int(np.random.uniform(
                self.audio_info['min_freq'] * 0.9,
                self.audio_info['min_freq'] * 1.1
            )),
            max_freq=int(np.random.uniform(
                self.audio_info['max_freq'] * 0.9,
                self.audio_info['max_freq'] * 1.1
            )),
            pitch_method=np.random.choice(['pyin', 'hybrid']),
            voicing_threshold=np.random.uniform(0.4, 0.8),
            confidence_weight=np.random.uniform(0.0, 1.0),
            min_note_duration=np.random.uniform(0.05, 0.3),
            max_gap=np.random.uniform(0.01, 0.1),
            onset_threshold=np.random.uniform(0.3, 0.8)
        )


def main():
    """Demonstrate the enhanced feedback system"""
    print("Enhanced MIDI Feedback System")
    print("=" * 60)

    # Initialize system
    system = EnhancedFeedbackSystem()

    # Show dataset analysis (if any)
    dataset_analysis = system.dataset_collector.analyze_dataset()
    if 'error' not in dataset_analysis:
        print(f"\nDataset Analysis:")
        print(f"  Samples: {dataset_analysis['n_samples']}")
        print(f"  Average score: {dataset_analysis['score_stats']['mean']:.4f}")
        print(f"  Best score: {dataset_analysis['score_stats']['max']:.4f}")

        if dataset_analysis['suggestions']:
            print(f"\nTop suggestions from data:")
            for suggestion in dataset_analysis['suggestions'][:3]:
                print(f"  • {suggestion}")
    else:
        print("\nNo dataset available yet. Run optimizations to build dataset.")

    # Show system capabilities
    print(f"\nSystem capabilities:")
    print(f"  • Parameter optimization with multiple strategies")
    print(f"  • Dataset collection and analysis")
    print(f"  • Adaptive parameter sampling")
    print(f"  • Recommendation generation")

    print(f"\nTo use:")
    print(f"  1. Prepare audio and reference MIDI pairs")
    print(f"  2. Call system.optimize_for_audio(audio_bytes, reference_midi_path)")
    print(f"  3. Use system.create_optimized_generator() for improved conversion")


if __name__ == "__main__":
    main()