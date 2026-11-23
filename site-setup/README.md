# Conference site setup

The site lives in `docs/` and is served by GitHub Pages. Emails are collected into a **private Google Sheet**
through a small Google Apps Script. Nothing about the emails is stored in this repo, and the script URL is the
only thing that goes into `docs/config.js`.

## 1. Create the private sheet and script (about 5 minutes)
1. In Google Sheets create a new blank sheet named e.g. "NeurIPS WiML leads". Do not share it.
2. Extensions > Apps Script. Replace the contents with `site-setup/Code.gs` and save.
3. Deploy > New deployment > type **Web app**. Execute as: **Me**. Who has access: **Anyone**. Deploy and
   authorize when asked.
4. Copy the web app URL (ends in `/exec`).
5. Paste it into `docs/config.js` as `endpoint`, commit, push.

Anyone who finds that URL can only *add* rows; they cannot read the sheet. Emails appear in the `Leads` tab.
After the conference, download the sheet as CSV and, if you like, turn the web app deployment off.

## 2. Turn on GitHub Pages
Repo Settings > Pages > Source: **Deploy from a branch**, branch `main`, folder `/docs`.

The site will be at `https://salmaroshdy-yg.github.io/NIPS-WiML-Verification-Without-Evidence/`.

## 3. The QR code
Point the QR code at the **connect page**:

`https://salmaroshdy-yg.github.io/NIPS-WiML-Verification-Without-Evidence/connect.html`

Visitors enter their email, then land on `index.html` (the paper page). "Skip" goes straight to the paper page.

## Test before the conference
Open the connect page on your phone, submit your own email, and check a row appears in the sheet.
If the connection drops mid-submit, the visitor still reaches the paper page and the email is retried on that page.
