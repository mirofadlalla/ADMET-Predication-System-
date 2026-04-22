# ============================================================
# Stage 1: Builder - Compile dependencies
# ============================================================
FROM python:3.11-slim as builder

# تثبيت أدوات البناء الضرورية
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# إنشاء البيئة الافتراضية
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# تحديث أدوات pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# خطوة هامة: تثبيت تورش نسخة الـ CPU أولاً 
# هذا يضمن أن chemprop عندما يبحث عن torch سيجده موجوداً بالفعل ولن يحاول تحميل نسخة الـ CUDA
RUN pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# تثبيت باقي المكتبات من مستودع PyPI الرسمي
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# تنظيف الملفات الزائدة لتقليل حجم الـ Image
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete

# ============================================================
# Stage 2: Runtime - Minimal production image
# ============================================================
FROM python:3.11-slim

# تثبيت مكتبات التشغيل فقط (ca-certificates ضرورية لطلبات الـ HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# نسخ البيئة الجاهزة من الـ builder
COPY --from=builder /opt/venv /opt/venv

# إعداد متغيرات البيئة
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# نسخ ملفات التطبيق (تأكد من وجود هذه المجلدات في جهازك)
COPY --chown=1000:1000 app/ ./app/
COPY --chown=1000:1000 models/ ./models/

# إنشاء مستخدم غير روت للأمان
RUN useradd -m -u 1000 admet && \
    chown -R admet:admet /app
USER admet

EXPOSE 8000

# اختبار الصحة (Health Check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# تشغيل التطبيق
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]