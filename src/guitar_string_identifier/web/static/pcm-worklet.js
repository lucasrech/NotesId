class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.chunkSize = opts.chunkSize || 2048;
    this.channelIndex = opts.channelIndex || 0;
    this.buffer = new Float32Array(this.chunkSize);
    this.offset = 0;

    this.port.onmessage = (event) => {
      const msg = event.data || {};
      if (typeof msg.channelIndex === "number") {
        this.channelIndex = msg.channelIndex;
      }
      if (typeof msg.chunkSize === "number" && msg.chunkSize >= 256) {
        this.chunkSize = msg.chunkSize;
        this.buffer = new Float32Array(this.chunkSize);
        this.offset = 0;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const chIdx = this.channelIndex < input.length ? this.channelIndex : 0;
    const chData = input[chIdx];
    if (!chData || chData.length === 0) return true;

    let readPos = 0;
    while (readPos < chData.length) {
      const remaining = this.chunkSize - this.offset;
      const toCopy = Math.min(remaining, chData.length - readPos);
      this.buffer.set(chData.subarray(readPos, readPos + toCopy), this.offset);
      this.offset += toCopy;
      readPos += toCopy;

      if (this.offset >= this.chunkSize) {
        const out = this.buffer;
        this.port.postMessage(out.buffer, [out.buffer]);
        this.buffer = new Float32Array(this.chunkSize);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
