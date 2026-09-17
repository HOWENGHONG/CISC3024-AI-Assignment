# Convert the report .docx to .pdf using Microsoft Word (Windows only).
#   powershell -ExecutionPolicy Bypass -File report\docx_to_pdf.ps1
param(
  [string]$Docx = (Join-Path $PSScriptRoot "CISC3024_AI_Assignment1_HoWengHong_uc325107.docx")
)
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
  $doc = $word.Documents.Open($Docx, $false, $true)
  $doc.Fields.Update() | Out-Null
  $doc.SaveAs2($Pdf, 17)   # 17 = wdFormatPDF
  $doc.Close($false)
  Write-Output "written $Pdf"
} finally {
  $word.Quit()
}
