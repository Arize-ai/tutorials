---
title: Limit failed sign-in attempts
expect: human
why: Authentication. Always high risk, and full of judgment calls (per account or per IP, lockout length, user enumeration through the error message) that a human should sign off.
---
Nothing stops someone from guessing passwords on the sign-in page as fast as they like.

After 5 failed attempts for an account in 15 minutes, further attempts for that account
should be refused for 15 minutes, even with the right password. The message shouldn't
reveal whether the account exists.
