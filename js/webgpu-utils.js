export async function checkWebGPUSupport() {
    if (!navigator.gpu) {
        return false;
    }
    return true;
}

export async function getGPUDevice() {
    if (!navigator.gpu) {
        throw new Error('WebGPU is not supported');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('Failed to get GPU adapter');
    }

    const device = await adapter.requestDevice();

    device.lost.then((info) => {
        console.error(`WebGPU device was lost: ${info.message}`);
        if (info.reason !== 'destroyed') {
            console.error('Device lost unexpectedly, may need to recreate');
        }
    });

    return { device, adapter };
}

export function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: usage,
        mappedAtCreation: true
    });

    if (data instanceof Float32Array) {
        new Float32Array(buffer.getMappedRange()).set(data);
    } else if (data instanceof Uint32Array) {
        new Uint32Array(buffer.getMappedRange()).set(data);
    } else if (data instanceof Int32Array) {
        new Int32Array(buffer.getMappedRange()).set(data);
    }

    buffer.unmap();
    return buffer;
}

export async function readBuffer(device, buffer, size) {
    const readBuffer = device.createBuffer({
        size: size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, size);
    device.queue.submit([commandEncoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    return result;
}
