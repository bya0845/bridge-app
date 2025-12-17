# PowerShell script to extract features and generate training data
# Run from project root: .\scripts\extract_and_generate.ps1

Write-Host "Extracting features from bridge data..." -ForegroundColor Cyan
python src/parser/extract_features.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Generating training data..." -ForegroundColor Cyan
    python src/parser/generate_training_set.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSuccess! Training data generated." -ForegroundColor Green
    } else {
        Write-Host "`nError generating training data!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`nError extracting features!" -ForegroundColor Red
    exit 1
}

