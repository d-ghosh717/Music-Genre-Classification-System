import React from 'react';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="space-y-6 max-w-5xl">
      {/* 1. Dataset */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md space-y-4 shadow-sm">
        <div>
          <h2 className="text-base font-semibold text-[#111827]">Dataset & Preprocessing</h2>
        </div>
        <p className="text-sm text-[#4B5563] leading-relaxed">
          The primary benchmark uses the canonical <strong>GTZAN Genre Dataset</strong>, consisting of 1,000 audio tracks across 10 distinct musical genres. All audio recordings are normalized to 22,050 Hz mono. To evaluate out-of-distribution generalisation, the models are also benchmarked on an independent 100-track evaluation set from <strong>MTG-Jamendo</strong> (10 tracks per genre across 100 distinct artists).
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-1">
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
            <span className="text-[#6B7280] block text-[11px]">Primary Dataset</span>
            <span className="font-semibold text-sm text-[#111827]">GTZAN (1,000 Tracks)</span>
          </div>
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
            <span className="text-[#6B7280] block text-[11px]">External Dataset</span>
            <span className="font-semibold text-sm text-[#111827]">MTG-Jamendo (100 Tracks)</span>
          </div>
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
            <span className="text-[#6B7280] block text-[11px]">Audio Format</span>
            <span className="font-semibold text-sm text-[#111827]">22,050 Hz Mono (30s)</span>
          </div>
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
            <span className="text-[#6B7280] block text-[11px]">Taxonomy (10 Classes)</span>
            <span className="font-semibold text-sm text-[#111827]">Blues to Rock</span>
          </div>
        </div>

        <div className="text-xs text-[#4B5563] space-y-1.5 pt-1">
          <div className="font-medium text-[#111827]">Genres Evaluated:</div>
          <div className="flex flex-wrap gap-1.5">
            {[
              'Blues',
              'Classical',
              'Country',
              'Disco',
              'Hiphop',
              'Jazz',
              'Metal',
              'Pop',
              'Reggae',
              'Rock',
            ].map((g) => (
              <span
                key={g}
                className="px-2.5 py-1 bg-[#F3F4F6] border border-[#E5E7EB] rounded text-[11px] text-[#374151]"
              >
                {g}
              </span>
            ))}
          </div>
          <p className="text-[11px] text-[#6B7280] pt-1">
            Partitioning uses stratified artist/song-level isolation into Training (70%), Validation (15%), and Test (15%) splits, ensuring zero excerpt leakage between partitions.
          </p>
        </div>
      </section>

      {/* 2. ANN Architecture */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md space-y-4 shadow-sm">
        <div>
          <h2 className="text-base font-semibold text-[#111827]">ANN Architecture (Multilayer Perceptron)</h2>
        </div>
        <p className="text-sm text-[#4B5563] leading-relaxed">
          The Artificial Neural Network operates on 75 extracted statistical acoustic features summarizing global spectral, timbral, rhythmic, and tonal properties across each full track. Features are normalized via a standard scaler fitted exclusively on the training partition.
        </p>

        <div className="p-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded text-xs font-mono text-[#111827] space-y-1 overflow-x-auto">
          <div className="font-medium text-[#374151]">Input: 75 Statistical Acoustic Features (MFCCs, Chroma, Centroid, Bandwidth, Rolloff, Contrast, RMS, Zero-Crossing Rate, Tempo)</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  ├─► Dense(384, ReLU) → BatchNormalization() → Dropout(0.35)</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  ├─► Dense(192, ReLU) → BatchNormalization() → Dropout(0.30)</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  ├─► Dense(96, ReLU) → Dropout(0.25)</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  └─► Dense(10, Softmax) ──► Output Genre Probabilities</div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-[#4B5563]">
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
            <span className="font-medium text-[#111827] block mb-1">Feature Representation</span>
            <span>Handcrafted statistical aggregations (mean & standard deviation) capturing long-term timbral and tonal envelopes.</span>
          </div>
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
            <span className="font-medium text-[#111827] block mb-1">Regularization</span>
            <span>Batch Normalization after linear projections, coupled with Dropout (0.25–0.35) to prevent feature co-adaptation.</span>
          </div>
        </div>
      </section>

      {/* 3. CNN Architecture */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md space-y-4 shadow-sm">
        <div>
          <h2 className="text-base font-semibold text-[#111827]">CNN Architecture (Convolutional Neural Network)</h2>
        </div>
        <p className="text-sm text-[#4B5563] leading-relaxed">
          The Convolutional Neural Network learns hierarchical 2D spatio-temporal representations directly from log-magnitude Mel-spectrogram patches. Audio is divided into 3-second segments with 50% overlap. During inference, segment-level probability distributions are averaged across the full recording.
        </p>

        <div className="p-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded text-xs font-mono text-[#111827] space-y-1 overflow-x-auto">
          <div className="font-medium text-[#374151]">Input: Log Mel-Spectrogram Patch (128 Mel Bands × 130 Time Steps × 1 Channel)</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  ├─► Conv2D(32, 3×3, ReLU) → BatchNormalization() → MaxPool2D(2×2) → Dropout(0.20)</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  ├─► Conv2D(64, 3×3, ReLU) → BatchNormalization() → MaxPool2D(2×2) → Dropout(0.25)</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  ├─► Conv2D(128, 3×3, ReLU) → BatchNormalization() → MaxPool2D(2×2) → Dropout(0.30)</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  ├─► Conv2D(256, 3×3, ReLU) → BatchNormalization() → GlobalAveragePooling2D()</div>
          <div className="text-[#9CA3AF]">  │</div>
          <div>  └─► Dense(128, ReLU) → Dropout(0.30) → Dense(10, Softmax) ──► Segment Probabilities</div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-[#4B5563]">
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
            <span className="font-medium text-[#111827] block mb-1">Spectrogram Parameters</span>
            <span>Sample Rate: 22,050 Hz, FFT Size: 2,048, Hop Length: 512, Mel Bands: 128, Frequency Range: 0–8,000 Hz.</span>
          </div>
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
            <span className="font-medium text-[#111827] block mb-1">Segment Aggregation</span>
            <span>Sliding window pooling averages softmax vectors over all 3-second segments to yield robust full-song consensus.</span>
          </div>
        </div>
      </section>

      {/* 4. Evaluation */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md space-y-4 shadow-sm">
        <div>
          <h2 className="text-base font-semibold text-[#111827]">Evaluation Metrics & Generalisation</h2>
        </div>
        <p className="text-sm text-[#4B5563] leading-relaxed">
          Model performance is evaluated across both in-domain test partitions (GTZAN) and out-of-distribution recordings (MTG-Jamendo) using standard classification metrics:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-[#4B5563]">
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded space-y-1">
            <span className="font-medium text-[#111827] block">Classification Accuracy & Macro F1</span>
            <p>
              Accuracy measures overall correct predictions. Macro-averaged F1 computes harmonic mean of precision and recall equally across all 10 genres, preventing majority-class dominance.
            </p>
          </div>
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded space-y-1">
            <span className="font-medium text-[#111827] block">Confusion Matrices</span>
            <p>
              Pairwise error matrices map true genres against predicted classes, exposing acoustic boundary ambiguities (e.g. Rock vs Metal, or Country vs Blues).
            </p>
          </div>
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded space-y-1 sm:col-span-2">
            <span className="font-medium text-[#111827] block">Out-of-Distribution Generalisation</span>
            <p>
              Evaluating independently sourced audio tests whether models have acquired generalized genre acoustics or merely overfit historical recording/mastering artifacts specific to the training corpus.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

