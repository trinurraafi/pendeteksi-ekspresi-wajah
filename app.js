// ======================================================================
// 1. PENGATURAN PROYEK
// ======================================================================
const CONFIG = {
    // Nama file model AI
    modelPath: './best.onnx',

    // Urutan HARUS sama dengan class di Roboflow
    labels: ["Sedih", "Marah"],

    // Confidence threshold
    threshold: 0.60,

    // IoU Threshold
    iouThreshold: 0.5
};

// ======================================================================
// 2. ELEMENT HTML
// ======================================================================
const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctxOverlay = overlay.getContext('2d');

const processor = document.getElementById('processor');
const ctxProcessor = processor.getContext('2d', {
    willReadFrequently: true
});

const status = document.getElementById('status');
const initBtn = document.getElementById('btn-init');

// ======================================================================
// 3. VARIABLE GLOBAL
// ======================================================================
let session;
const TARGET_SIZE = 640;

// ======================================================================
// 4. LOAD MODEL AI
// ======================================================================
initBtn.addEventListener('click', async () => {

    initBtn.disabled = true;
    initBtn.innerText = "MEMUAT MODEL AI...";

    try {

        ort.env.wasm.wasmPaths =
            'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

        session = await ort.InferenceSession.create(
            CONFIG.modelPath,
            {
                executionProviders: ['webgl', 'wasm']
            }
        );

        startCamera();

    } catch (e) {

        console.error(e);

        status.innerText =
            "GAGAL MEMUAT MODEL AI";

        initBtn.disabled = false;
        initBtn.innerText = "COBA LAGI";
    }
});

// ======================================================================
// 5. START CAMERA
// ======================================================================
async function startCamera() {

    const stream =
        await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480
            },
            audio: false
        });

    video.srcObject = stream;

    video.onloadedmetadata = () => {

        video.play();

        status.innerText =
            "SISTEM AKTIF";

        initBtn.style.display = "none";

        requestAnimationFrame(processFrame);
    };
}

// ======================================================================
// 6. PROSES DETEKSI AI
// ======================================================================
async function processFrame() {

    if (!session) return;

    // =========================================================
    // AMBIL FRAME VIDEO
    // =========================================================
    ctxProcessor.drawImage(
        video,
        0,
        0,
        TARGET_SIZE,
        TARGET_SIZE
    );

    const imageData =
        ctxProcessor.getImageData(
            0,
            0,
            TARGET_SIZE,
            TARGET_SIZE
        ).data;

    const float32Data =
        new Float32Array(
            3 * TARGET_SIZE * TARGET_SIZE
        );

    // =========================================================
    // CONVERT RGB
    // =========================================================
    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {

        float32Data[i] =
            imageData[i * 4] / 255.0;

        float32Data[i + TARGET_SIZE * TARGET_SIZE] =
            imageData[i * 4 + 1] / 255.0;

        float32Data[i + 2 * TARGET_SIZE * TARGET_SIZE] =
            imageData[i * 4 + 2] / 255.0;
    }

    // =========================================================
    // INPUT KE AI
    // =========================================================
    const inputTensor =
        new ort.Tensor(
            'float32',
            float32Data,
            [1, 3, TARGET_SIZE, TARGET_SIZE]
        );

    const results =
        await session.run({
            [session.inputNames[0]]: inputTensor
        });

    const output =
        results[session.outputNames[0]].data;

    // =========================================================
    // PARSE OUTPUT AI
    // =========================================================
    const numClasses = CONFIG.labels.length;
    const elements = 8400;

    let rawBoxes = [];

    for (let i = 0; i < elements; i++) {

        let maxScore = 0;
        let classId = -1;

        // Cari score tertinggi
        for (let c = 0; c < numClasses; c++) {

            const score =
                output[i + (4 + c) * elements];

            if (score > maxScore) {

                maxScore = score;
                classId = c;
            }
        }

        // Jika lolos threshold
        if (maxScore > CONFIG.threshold) {

            let x = output[i];
            let y = output[i + elements];
            let w = output[i + 2 * elements];
            let h = output[i + 3 * elements];

            // Normalisasi ukuran
            if (w <= 1.5) {

                x *= TARGET_SIZE;
                y *= TARGET_SIZE;
                w *= TARGET_SIZE;
                h *= TARGET_SIZE;
            }

            rawBoxes.push({
                x: x - w / 2,
                y: y - h / 2,
                w: w,
                h: h,
                score: maxScore,
                classId: classId
            });
        }
    }

    // =========================================================
    // NON MAX SUPPRESSION
    // =========================================================
    const finalBoxes =
        nonMaxSuppression(
            rawBoxes,
            CONFIG.iouThreshold
        );

    // =========================================================
    // DRAW RESULT
    // =========================================================
    drawBoxes(finalBoxes);

    requestAnimationFrame(processFrame);
}

// ======================================================================
// 7. IOU
// ======================================================================
function calculateIoU(box1, box2) {

    const xA =
        Math.max(box1.x, box2.x);

    const yA =
        Math.max(box1.y, box2.y);

    const xB =
        Math.min(
            box1.x + box1.w,
            box2.x + box2.w
        );

    const yB =
        Math.min(
            box1.y + box1.h,
            box2.y + box2.h
        );

    const intersectionArea =
        Math.max(0, xB - xA) *
        Math.max(0, yB - yA);

    return (
        intersectionArea /
        (
            (box1.w * box1.h) +
            (box2.w * box2.h) -
            intersectionArea
        )
    );
}

// ======================================================================
// 8. NON MAX SUPPRESSION
// ======================================================================
function nonMaxSuppression(boxes, iouThreshold) {

    boxes.sort((a, b) => b.score - a.score);

    const result = [];

    while (boxes.length > 0) {

        const current = boxes.shift();

        result.push(current);

        boxes = boxes.filter(box =>
            calculateIoU(current, box) < iouThreshold
        );
    }

    return result;
}

// ======================================================================
// 9. DRAW BOXES (VERSI KOTAK DIPERKECIL)
// ======================================================================
function drawBoxes(boxes) {

    ctxOverlay.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    boxes.forEach(box => {

        const scaleX =
            overlay.width / TARGET_SIZE;

        const scaleY =
            overlay.height / TARGET_SIZE;

        // =====================================================
        // SHRINK BOX
        // =====================================================
        const shrinkFactor = 0.65;

        const newW =
            box.w * shrinkFactor;

        const newH =
            box.h * shrinkFactor;

        const newX =
            box.x + (box.w - newW) / 2;

        const newY =
            box.y + (box.h - newH) / 2;

        // =====================================================
        // BOX
        // =====================================================
        ctxOverlay.strokeStyle = "#34C759";
        ctxOverlay.lineWidth = 3;

        ctxOverlay.strokeRect(
            newX * scaleX,
            newY * scaleY,
            newW * scaleX,
            newH * scaleY
        );

        // =====================================================
        // BACKGROUND LABEL
        // =====================================================
        const label =
            `${CONFIG.labels[box.classId]} ${(box.score * 100).toFixed(0)}%`;

        ctxOverlay.font =
            "bold 16px Arial";

        const textWidth =
            ctxOverlay.measureText(label).width;

        ctxOverlay.fillStyle =
            "#34C759";

        ctxOverlay.fillRect(
            newX * scaleX,
            (newY * scaleY) - 25,
            textWidth + 10,
            22
        );

        // =====================================================
        // TEXT LABEL
        // =====================================================
        ctxOverlay.fillStyle = "#FFFFFF";

        ctxOverlay.fillText(
            label,
            (newX * scaleX) + 5,
            (newY * scaleY) - 8
        );
    });
}
