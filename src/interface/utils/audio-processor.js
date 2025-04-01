class AudioProcessor extends AudioWorkletProcessor {
	// Buffer audio chunks until we have a desired size
	_bufferSize = 2048 // Send chunks of this size (adjust as needed)
	_bytesWritten = 0
	_buffer = new Float32Array(this._bufferSize)

	constructor() {
		super()
		console.log("[AudioProcessor] Worklet created.")
	}

	process(inputs, outputs, parameters) {
		// inputs[0][0] is the Float32Array of PCM data for the first channel
		const inputChannelData = inputs[0][0]

		if (!inputChannelData) {
			// No input data, maybe mic stopped?
			return true // Keep processor alive
		}

		// Append new data to buffer
		const dataToCopy = Math.min(
			this._bufferSize - this._bytesWritten,
			inputChannelData.length
		)
		this._buffer.set(
			inputChannelData.subarray(0, dataToCopy),
			this._bytesWritten
		)
		this._bytesWritten += dataToCopy

		// If buffer is full, send it to the main thread
		if (this._bytesWritten >= this._bufferSize) {
			// Send a *copy* of the buffer
			this.port.postMessage(this._buffer.slice(0))

			// Reset buffer and handle leftover data
			const leftoverData = inputChannelData.subarray(dataToCopy)
			this._buffer.fill(0) // Clear buffer
			this._bytesWritten = leftoverData.length
			if (this._bytesWritten > 0) {
				this._buffer.set(leftoverData)
			}
		}

		// Return true to keep the processor alive
		return true
	}
}

registerProcessor("audio-processor", AudioProcessor)
