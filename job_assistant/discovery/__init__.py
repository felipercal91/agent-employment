"""Job discovery from sources that do not involve automating job portals.

- Inbox: postings the user saved by hand into data/inbox/.
- Company career boards through the public, read-only JSON APIs that ATS vendors
  (Greenhouse, Lever, Ashby) publish for embedding job lists. No accounts, no logins,
  no form submission, one request per company per run.
LinkedIn/Indeed are never contacted.
"""
