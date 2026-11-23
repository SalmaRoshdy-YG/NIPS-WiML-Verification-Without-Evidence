/**
 * Private email collector for the conference site.
 * Paste into Extensions > Apps Script of a Google Sheet that only you can open.
 * Deploy as a Web app: Execute as "Me", Who has access "Anyone".
 * Emails land in the "Leads" tab of that private sheet, never in the repo.
 */
var SHEET_NAME = "Leads";

function doPost(e) {
  var lock = LockService.getScriptLock();
  try {
    lock.waitLock(10000);
    var d = JSON.parse(e.postData.contents);
    var email = clean_(d.email, 254);
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return ok_();
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sh = ss.getSheetByName(SHEET_NAME) || ss.insertSheet(SHEET_NAME);
    if (sh.getLastRow() === 0) sh.appendRow(["Received", "Name", "Email", "Source"]);
    sh.appendRow([new Date(), clean_(d.name, 100), email, clean_(d.source, 60)]);
  } catch (err) {
    // swallow: the visitor should still reach the paper page
  } finally {
    try { lock.releaseLock(); } catch (x) {}
  }
  return ok_();
}

function doGet() { return ok_(); }

function ok_() { return ContentService.createTextOutput("ok"); }

// Trim, cap length, and stop spreadsheet formula injection (=, +, -, @).
function clean_(v, max) {
  var s = String(v == null ? "" : v).trim().slice(0, max);
  return /^[=+\-@]/.test(s) ? "'" + s : s;
}
