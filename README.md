# SaveStack - CONTENT HUB FOR ALL🚀

> **Save it. Sort it. Actually use it.**

SaveStack is a lightweight, privacy-first web app that helps users save important links, notes, and resources in one place — and gently nudges them to come back before their saves disappear.

**I’ve built an AI-powered tool that organizes all the educational content students save across apps like Instagram, WhatsApp, YouTube, and LinkedIn.
The tool automatically categorizes posts by domain (like full-stack, UI/UX, game dev), summarizes them, and marks content as “viewed” or “done.”
It solves a real problem: students save hundreds of reels and posts but forget about them within days. 
My tool becomes a smart “content brain” that filters, reminds, sorts, and creates a personalized learning playlist.**

Unlike traditional bookmark managers that become digital graveyards, **SaveStack is built around recall, revisits, and intentional retention**.

---

## Navigating the Interface

![Image](https://github.com/user-attachments/assets/a710a3b5-7110-4c07-81d1-c22161337496)

![Image](https://github.com/user-attachments/assets/1270050b-110b-459e-9d1d-ddf7e9003fcb)


---

## ✨ Core Idea

Most people save things with good intentions… and never return.

SaveStack flips this behavior by:

* Encouraging **periodic revisits**
* Keeping storage **lean and intentional**
* Using **reminders instead of hoarding**

Think of it as a **temporary brain cache** rather than infinite storage.

---

## 🔑 Key Features

### 📌 Save Anything

* Save links, notes, and references instantly
* Minimal friction, fast UI

### ⏳ Time-Aware Storage

* Saves are **auto-deleted after 30 days** (Pending)
* Keeps IndexedDB storage clean and intentional (Enables Local Storage)

### 🔔 Smart Email Reminders (Pending)

* Reminder emails sent every **15–30 days** 
* Notifies users before their saves expire

### 🔐 Auth & Profiles

* Secure authentication powered by **Supabase Auth**
* User profiles with activity tracking

### ⚡ Lightweight by Design

* No forced subscriptions
* No dark patterns
* No infinite clutter

---

## 🧠 Philosophy (Why SaveStack?)

* ❌ Not another bookmark dump
* ❌ Not a bloated second brain
* ✅ A **focused recall tool**
* ✅ A **learning companion**

SaveStack helps you *remember what you saved* — or lets it go.

---

## How do 'Streaks' work?

![Image](https://github.com/user-attachments/assets/f7027437-8f8e-468d-a6f9-e50919828617)

---

## StashCast - For people on the go, all the time.

![Image](https://github.com/user-attachments/assets/5717c51b-d0c2-4438-afd4-64ad49cb6cdf)

---

## 🛠 Tech Stack

### Frontend

* React
* Typescript
* Tailwind CSS
* HTML/CSS
* IndexedDB for local-first storage

### Backend

* **Supabase**

  * Auth
  * Postgres database
  * Edge Functions

### Email & Automation (Yet to Implement)

* Supabase Edge Functions (Deno)
* External Cron (GitHub / cron-job.org)
* Email provider (e.g. Resend)

---

## 🧩 Architecture Overview

```text
User → SaveStack Web App
     → IndexedDB (local saves)
     → Supabase Auth & Profiles
     → Edge Function (email reminders)
     → Email Service → User Inbox
```

---

## 🔄 Reminder System Logic

1. User saves content
2. Save timestamp is recorded
3. Cron triggers Edge Function periodically
4. Inactive users are queried
5. Reminder email is sent (Once every 15/30 days - If found inactive)
6. After 30 days → saves are cleared (Pending)

---

## 🧪 Project Status

* ✅ Core save & recall flow
* ✅ Supabase auth integration
* ✅ Email reminder system
* 🚧 UX polish
* 🚧 Analytics & insights

---

## 🧭 Future Ideas

* Daily / weekly recap emails
* "Revive or Delete" actions
* Modelling a Browser extension
* Mobile PWA enhancements

---

## 🧑‍💻 Author

**Shreyas S**
Student Developer | Web • Cloud • AI

Built with curiosity, frustration with bookmarks, and a love for clean systems.

---

## 📄 License

MIT License

Copyright (c) 2025 Shreyas S

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

> *Save intentionally. Revisit consciously. Let go freely.*