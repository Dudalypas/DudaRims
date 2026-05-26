$ErrorActionPreference = "Continue"
$env:JAVA_TOOL_OPTIONS = "--enable-native-access=ALL-UNNAMED"

$flow = ".maestro/fullFlow.yaml"
$iterations = 10
$resultsDir = ".maestro/results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$log = "$resultsDir/stability_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
$passed = 0

@"
Maestro stabilumo testas pradetas: $(Get-Date)
Scenarijus: $flow
Iteracijos: $iterations

"@ | Set-Content -Path $log -Encoding utf8

for ($i = 1; $i -le $iterations; $i++) {
    Write-Host "Vykdoma iteracija $i is $iterations..."

    & maestro test $flow *> $null
    $exitCode = $LASTEXITCODE

    if ($exitCode -eq 0) {
        $passed++
        "Iteracija ${i}: IVYKDYTA" | Add-Content -Path $log -Encoding utf8
        Write-Host "Iteracija $i IVYKDYTA"
    } else {
        "Iteracija ${i}: NEIVYKDYTA, exitCode=$exitCode" | Add-Content -Path $log -Encoding utf8
        "Santrauka: ivykdyta=$passed, neivykdyta=1, is viso=$iterations" | Add-Content -Path $log -Encoding utf8
        Write-Host "Iteracija $i NEIVYKDYTA"
        exit 1
    }
}

@"

Maestro stabilumo testas uzbaigtas: $(Get-Date)
Santrauka: ivykdyta=$passed, neivykdyta=0, is viso=$iterations
Visos $iterations iteracijos uzbaigtos sekmingai.
"@ | Add-Content -Path $log -Encoding utf8

Write-Host "Visos $iterations iteracijos uzbaigtos sekmingai."
Write-Host "Log failas: $log"
exit 0