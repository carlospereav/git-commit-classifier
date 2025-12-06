# 🚀 Deployment Guide - Cloud Run

Esta guía te lleva paso a paso para desplegar el Git Commit Classifier en Google Cloud Run.

---

## 📋 Prerrequisitos

1. **Cuenta de Google Cloud** con billing activado
2. **Google Cloud CLI** instalado: [Instrucciones](https://cloud.google.com/sdk/docs/install)
3. **Docker** instalado (opcional, para pruebas locales)

---

## 🔧 Configuración Inicial (Una sola vez)

### 1. Autenticarse en GCP

```bash
gcloud auth login
```

### 2. Crear o seleccionar proyecto

```bash
# Crear nuevo proyecto
gcloud projects create git-commit-classifier --name="Git Commit Classifier"

# O seleccionar existente
gcloud config set project TU_PROJECT_ID
```

### 3. Habilitar APIs necesarias

```bash
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com
```

### 4. Configurar región por defecto

```bash
gcloud config set run/region us-central1
```

---

## 🐳 Opción A: Despliegue Manual (Rápido)

### Paso 1: Build y push de la imagen

```bash
# Desde la raíz del proyecto
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/git-commit-classifier
```

### Paso 2: Deploy a Cloud Run

```bash
gcloud run deploy git-commit-classifier \
    --image gcr.io/$(gcloud config get-value project)/git-commit-classifier \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 3 \
    --allow-unauthenticated
```

### Paso 3: ¡Listo!

Cloud Run te dará una URL como:
```
https://git-commit-classifier-xxxxx-uc.a.run.app
```

---

## 🔄 Opción B: CI/CD Automático con Cloud Build

Esta opción despliega automáticamente cada vez que haces push a GitHub.

### Paso 1: Conectar repositorio

1. Ve a [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers)
2. Click en "Connect Repository"
3. Selecciona "GitHub" y autoriza
4. Elige tu repositorio `git-commit-classifier`

### Paso 2: Crear Trigger

1. Click en "Create Trigger"
2. Configura:
   - **Name:** `deploy-on-push`
   - **Event:** Push to branch
   - **Branch:** `^main$`
   - **Configuration:** Cloud Build configuration file
   - **Location:** `/cloudbuild.yaml`
3. Click "Create"

### Paso 3: Dar permisos a Cloud Build

```bash
# Obtener el número del proyecto
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')

# Dar permisos de Cloud Run Admin
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/run.admin"

# Dar permisos de Service Account User
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"
```

### Paso 4: ¡Push y listo!

```bash
git add .
git commit -m "feat: add Cloud Run deployment"
git push origin main
```

Cloud Build construirá y desplegará automáticamente.

---

## 🧪 Pruebas Locales con Docker

Antes de desplegar, puedes probar localmente:

```bash
# Build
docker build -t commit-classifier .

# Run
docker run -p 8080:8080 commit-classifier

# Abrir en navegador
# http://localhost:8080
```

---

## 💰 Estimación de Costos

| Concepto | Costo |
|----------|-------|
| **Cloud Run** | $0.00002400/vCPU-second + $0.00000250/GiB-second |
| **Free Tier** | 2 millones de requests/mes, 360,000 GiB-seconds, 180,000 vCPU-seconds |
| **Container Registry** | $0.026/GB/mes (después de 0.5GB gratis) |

**Estimación para uso moderado (1000 requests/día):**
- Con cold starts frecuentes: ~$5-10/mes
- Con `min-instances=1`: ~$25-30/mes (sin cold starts)

---

## 🔧 Configuración Avanzada

### Reducir Cold Starts

```bash
# Mantener 1 instancia siempre activa (aumenta costo ~$25/mes)
gcloud run services update git-commit-classifier --min-instances 1
```

### Dominio Personalizado

```bash
# Mapear dominio personalizado
gcloud run domain-mappings create \
    --service git-commit-classifier \
    --domain commits.tudominio.com
```

### Variables de Entorno

```bash
# Agregar GitHub token para más API calls
gcloud run services update git-commit-classifier \
    --set-env-vars "GITHUB_TOKEN=ghp_xxxx"
```

---

## 🐛 Troubleshooting

### Error: "Container failed to start"

```bash
# Ver logs
gcloud run services logs read git-commit-classifier --limit 50
```

### Error: "Memory limit exceeded"

```bash
# Aumentar memoria
gcloud run services update git-commit-classifier --memory 4Gi
```

### Cold start muy lento

El modelo de ~255MB tarda en cargar. Opciones:
1. Usar `min-instances=1` ($$)
2. Convertir modelo a ONNX (más rápido)
3. Usar modelo más pequeño

---

## 📊 Monitoreo

- **Cloud Run Console:** https://console.cloud.google.com/run
- **Logs:** https://console.cloud.google.com/logs
- **Métricas:** Request count, latency, memory usage

---

¡Happy deploying! 🎉

