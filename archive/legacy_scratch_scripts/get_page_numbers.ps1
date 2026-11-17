$word = New-Object -ComObject Word.Application
$word.Visible = $true

$docPath = "d:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\RoadSense_Report_Final_Paged.docx"
$doc = $word.Documents.Open($docPath)
$word.ActiveWindow.View.Type = 3 # wdPrintView

Start-Sleep -Seconds 2

$headings = @(
    "1.1 INTRODUCTION", "1.2 AIM AND OBJECTIVES", "1.3 PROBLEM STATEMENT", "1.4 SCOPE OF THE PROJECT", "1.5 CHALLENGES",
    "2.1 LITERATURE SURVEY", "2.2 REQUIREMENTS",
    "3.1 PROPOSED METHODOLOGY", "3.2 DATASETS", "3.3 FLOW OF THE MODEL",
    "4.1 EXPERIMENTAL SETUP", "4.2 RESULT SUMMARY TABLES", "4.3 ANALYSIS OF RESULTS", "4.4 COMPARISON WITH EXISTING METHODS",
    "5.1 KEY FINDINGS", "5.2 INTERPRETATION OF RESULTS", "5.3 IMPLICATIONS OF THE STUDY", "5.4 FULFILLMENT OF OBJECTIVES",
    "6.1 RESULTS", "6.2 FUTURE WORK", "6.3 REFERENCES"
)

$results = @{}

foreach ($h in $headings) {
    $selection = $word.Selection
    $selection.HomeKey(6) > $null
    $find = $selection.Find
    $find.Text = $h
    $find.MatchCase = $true
    if ($find.Execute()) {
        $adjPageNum = $selection.Information(1)
        $results[$h] = $adjPageNum
    }
}

$doc.Close(0)
$word.Quit()

$results | ConvertTo-Json | Out-File "d:\Recovered_From_Sukee\Downloads\Saket\COLLEGE\Road Health\main\page_numbers.json"
