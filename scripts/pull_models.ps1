# Idempotent model pull for Windows.
$ErrorActionPreference = "Stop"

$models = @("qwen2.5vl:7b", "llama3.1:8b", "nomic-embed-text")
$installed = (ollama list) -split "`n" | Select-Object -Skip 1 | ForEach-Object { ($_ -split "\s+")[0] }

foreach ($m in $models) {
    if ($installed -contains $m) {
        Write-Host "[skip] $m already present"
    } else {
        Write-Host "[pull] $m"
        ollama pull $m
    }
}

Write-Host "All required models are available."
