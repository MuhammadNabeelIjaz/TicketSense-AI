# Documentation Index

Overview of all planning documents, templates, and reference materials available in `docs/`.

---

## Document Catalog

| Directory | File | Type | Purpose & Editing Rule |
|---|---|---|---|
| [`01_Project_Brief/`](01_Project_Brief/) | [`Project_2_Support_Ticket_Classifier.pdf`](01_Project_Brief/Project_2_Support_Ticket_Classifier.pdf) | PDF | Official project brief, functional requirements, and project scope. **Read only.** |
| [`02_Research_and_Planning/`](02_Research_and_Planning/) | [`Research_Planning_Template.html`](02_Research_and_Planning/Research_Planning_Template.html)<br>[`Research_Planning_Template.pdf`](02_Research_and_Planning/Research_Planning_Template.pdf) | HTML (Master)<br>PDF (Print) | **Doc 1**: 13 research areas (Blocks A–H), decision register, and two-checkpoint readiness gate (40 pages). **Edit the HTML.** |
| [`03_Technical_Learning/`](03_Technical_Learning/) | [`Technical_Learning_Roadmap.html`](03_Technical_Learning/Technical_Learning_Roadmap.html)<br>[`Technical_Learning_Roadmap.pdf`](03_Technical_Learning/Technical_Learning_Roadmap.pdf) | HTML (Master)<br>PDF (Print) | **Doc 2**: Technical curriculum covering 65 concepts across 9 units (A–I) in prerequisite order (16 pages). **Read only.** |
| [`04_Master_Journey_Tracker/`](04_Master_Journey_Tracker/) | [`Master_Journey_Tracker.html`](04_Master_Journey_Tracker/Master_Journey_Tracker.html)<br>[`Master_Journey_Tracker.pdf`](04_Master_Journey_Tracker/Master_Journey_Tracker.pdf) | HTML (Master)<br>PDF (Print) | **Doc 3**: Execution tracker with 51 tasks, 29 concept rows, schedule planner, and handover prompt (15 pages). **Edit the HTML.** |
| [`05_References_and_Resources/`](05_References_and_Resources/) | `.gitkeep` | Directory | Reserved for external references, industry standards, and research artifacts. |

---

## How the Documents Fit Together

- **Doc 1: Research & Planning Template** defines *what* to investigate and holds findings, technical decisions, and architecture choices.
- **Doc 2: Technical Learning Roadmap** explains the underlying ML and engineering concepts (65 concepts across Units A–I). Holds no status.
- **Doc 3: Master Journey Tracker** is the **single source of truth for progress**. Holds all 51 task statuses, 29 concept rows, daily logs, and evidence.

```
       ┌────────────────────────────────────────────────────────┐
       │                  01_Project_Brief                      │
       │     Project_2_Support_Ticket_Classifier.pdf            │
       │              (Requirements & Scope)                    │
       └───────────────────────────┬────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────┐
│     02_Research_and_Planning    │ │      03_Technical_Learning      │
│  Doc 1: Research & Decisions    │ │   Doc 2: Learning Curriculum    │
│ (Research_Planning_Template)    │ │   (Technical_Learning_Roadmap)  │
└────────────────┬────────────────┘ └────────────────┬────────────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │              04_Master_Journey_Tracker                 │
       │             Doc 3: Master Journey Tracker              │  ◄── Status, Daily Logs
       │               (Master_Journey_Tracker)                 │      & Evidence Recorded
       │            *Single Source of Truth for Progress*       │      Here ONLY
       └────────────────────────────────────────────────────────┘
```

### Core Working Rules

1. **Progress**: Recorded **only** in Doc 3 (`Master_Journey_Tracker.html`).
2. **Findings & Decisions**: Recorded **only** in Doc 1 (`Research_Planning_Template.html`).
3. **Concept Meanings**: Explained **only** in Doc 2 (`Technical_Learning_Roadmap.html`).
4. **Editable Masters**: The `.html` file is the editable workspace for each document; the accompanying `.pdf` is a fixed print copy (never edit the PDF).
5. **Persistence**: LocalStorage autosaves in your browser; after every working session click **Save a copy** and keep that file.

---

## Terminology Glossary

| Term | Definition | Primary Location |
|---|---|---|
| **Phases 1–9** | Sequential project delivery phases: Onboarding (1), Research (2), Technical Learning (3), Architecture & Design (4), Model & App Development (5), Testing & Evaluation (6), Optimization & Review (7), Final Deliverables (8), Handover & Demo (9). | Tracker Section 6 |
| **Tasks (1.1 to 9.2)** | 51 discrete work items indexed by phase number (42 Must, 5 Should, 4 Could). | Tracker Section 5 |
| **Days 1–7 / Days 1–14** | Target 7-day delivery plan (~7 hrs/day) or 14-day realistic schedule (~3.5 hrs/day, recommended default). | Tracker Section 7 |
| **Areas 1–13** | Research topics grouped into Blocks A through H in Doc 1. | Doc 1 (Areas 1–13) |
| **Units A–I (Concept A1..)** | 65 technical concepts (42 Essential, 19 Useful, 4 Advanced) arranged in prerequisite order. | Doc 2 (Units A–I) |
| **Checkpoints 1 & 2** | Two-checkpoint readiness gate: Checkpoint 1 before model training (Task 2.12); Checkpoint 2 before app development (Task 2.13). | Doc 1 Readiness Gate; Tracker 2.12 & 2.13 |
| **Must / Should / Could** | MoSCoW prioritization scheme defining Minimum Path (42 Must tasks) vs. Full Track scope. | Tracker Section 4 |
