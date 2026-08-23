# Nyaya — human verification packet

Two acceptance criteria are reserved for you. No agent ticked them.

---

## A. Five sections, word for word against `data/raw/a202345.pdf`

SPEC.md section 7 acceptance criterion. Open the PDF at each printed page and compare.

### BNS 271 — Negligent act likely to spread infection of disease dangerous to life
*PDF page 83 · chapter XV OF OFFENCES AFFECTING THE PUBLIC HEALTH, SAFETY, CONVENIENCE, DECENCY AND MORALS · 380 chars · 0 illustrations*

```
271. Negligent act likely to spread infection of disease dangerous to life.—Whoever unlawfully or 
negligently does any act which is, and which he knows or has reason to believe to be, likely to spread the 
infection of any disease dangerous to life, shall be punished with imprisonment of either description for a 
term which may extend to six months, or with fine, or with both.
```

### BNS 55 — Abetment of offence punishable with death or imprisonment for life
*PDF page 33 · chapter IV OF ABETMENT, CRIMINAL CONSPIRACY AND ATTEMPT · 1168 chars · 1 illustrations*

```
55. Abetment of offence punishable with death or imprisonment for life.—Whoever abets the 
commission of an offence punishable with death or imprisonment for life, shall, if that offence be not 
committed in consequence of the abetment, and no express provision is made under this Sanhita for the 
punishment of such abetment, be punished with imprisonment of either description for a term which may 
extend to seven years, and shall also be liable to fine; and if any act for which the abettor is liable in 
consequence of the abetment, and which causes hurt to any person, is done, the abettor shall be liable to 
imprisonment of either description for a term which may extend to fourteen years, and shall also be liable 
to fine. 
Illustration. 
A instigates B to murder Z. The offence is not committed. If B had murdered Z, he would have been 
subject to the punishment of death or imprisonment for life. Therefore, A is liable to imprisonment for a 
term which may extend to seven years and also to a fine; and if any hurt be done to Z in consequence of 
the abetment, he will be liable to imprisonment for a term which may extend to fourteen years, and to 
fine.
```

### BNS 66 — Punishment for causing death or resulting in persistent vegetative state of victim
*PDF page 38 · chapter V OF OFFENCES AGAINST WOMAN AND CHILD · 578 chars · 0 illustrations*

```
66. Punishment for causing death or resulting in persistent vegetative state of victim.—
Whoever, commits an offence punishable under sub-section (1) or sub-section (2) of section 64 and in the 
course of such commission inflicts an injury which causes the death of the woman or causes the woman 
to be in a persistent vegetative state, shall be punished with rigorous imprisonment for a term which shall 
not be less than twenty years, but which may extend to imprisonment for life, which shall mean 
imprisonment for the remainder of that person’s natural life, or with death.
```

### BNS 293 — Continuance of nuisance after injunction to discontinue
*PDF page 86 · chapter XV OF OFFENCES AFFECTING THE PUBLIC HEALTH, SAFETY, CONVENIENCE, DECENCY AND MORALS · 400 chars · 0 illustrations*

```
293. Continuance of nuisance after injunction to discontinue.—Whoever repeats or continues a 
public nuisance, having been enjoined by any public servant who has lawful authority to issue such 
injunction not to repeat or continue such nuisance, shall be punished with simple imprisonment for a term 
which may extend to six months, or with fine which may extend to five thousand rupees or with both.
```

### BNS 358 — Repeal and savings
*PDF page 110 · chapter XX REPEAL AND SAVINGS · 4176 chars · 0 illustrations*

```
358. Repeal and savings.—(1) The Indian Penal Code (45 of 1860) is hereby repealed. 
(2) Notwithstanding the repeal of the Code referred to in sub-section (1), it shall not affect,— 
(a) the previous operation of the Code so repealed or anything duly done or suffered thereunder; 
or 
(b) any right, privilege, obligation or liability acquired, accrued or incurred under the Code so 
repealed; or 
(c) any penalty, or punishment incurred in respect of any offences committed against the Code so 
repealed; or 
(d) any investigation or remedy in respect of any such penalty, or punishment; or 
(e) any proceeding, investigation or remedy in respect of any such penalty or punishment as 
aforesaid, and any such proceeding or remedy may be instituted, continued or enforced, and any such 
penalty may be imposed as if that Code had not been repealed. 
(3) Notwithstanding such repeal, anything done or any action taken under the said Code shall be 
deemed to have been done or taken under the corresponding provisions of this Sanhita. 
(4) The mention of particular matters in sub-section (2) shall not be held to prejudice or affect the 
general application of section 6 of the General Clauses Act, 1897 (10 of 1897) with regard to the effect of 
the repeal. 
STATEMENT OF OBJECTS AND REASONS 
In the year 1834, the first Indian Law Commission was constituted under the Chairman-ship of Lord 
Thomas Babington Macaulay to examine the jurisdiction, power and rules of the existing Courts as well 
as the police establishments and the laws in force in India. 
2. The Commission suggested various enactments to the Government. One of the important 
recommendations made by the Commission was on, Indian Penal Code which was enacted in 1860 and 
the said Code is still continuing in the country with some 
[... truncated, full text in data/processed/sections.json]
```

---

## B. Twenty mappings against NCRB `data/raw/BNS2023.pdf`, pages 20-73

SPEC.md section 10 acceptance criterion.

| BNS | Section title | Maps to IPC |
|---|---|---|
| 1 | Short title, commencement and application | 1, 2, 3, 4, 5 |
| 2 | Definitions | 33, 47, 28, 50, 41, 30, 48, 39, 31, 10, 23 |
| 65 | Punishment for rape in certain cases | 376, 376AB |
| 70 | Gang rape | 376D, 376DA, 376DB |
| 73 | Printing or publishing any matter relating to Court  | 228A |
| 101 | Murder | 300 |
| 103 | Punishment for murder | 302 |
| 111 | Organised crime | **(none — expected: New Section)** |
| 115 | Voluntarily causing hurt | 321, 323 |
| 177 | Failure to keep election accounts | 171-I |
| 196 | Promoting enmity between different groups on grounds | 153A, 153AA |
| 303 | Theft | 378, 379 |
| 304 | Snatching | **(none — expected: New Section)** |
| 316 | Criminal breach of trust | 405, 406, 407, 408, 409 |
| 318 | Cheating | 415, 417, 418, 420 |
| 319 | Cheating by personation | 416, 419 |
| 324 | Mischief | 425, 426, 427, 440 |
| 351 | Criminal intimidation | 503, 506, 507 |
| 356 | Defamation | 499, 500, 501, 502 |
| 358 | Repeal and savings | **(none — expected: New Section)** |

### Rows to check hardest

These five were silently wrong until a review round caught them. Two of them
(70 and 196) looked complete while missing citations, so they are the best test
of whether the parser is now right:

- **BNS 65** → 376, 376AB
- **BNS 70** → 376D, 376DA, 376DB
- **BNS 73** → 228A
- **BNS 177** → 171-I
- **BNS 196** → 153A, 153AA

### Sections with no IPC ancestor

10 sections: 48, 69, 95, 111, 112, 113, 152, 226, 304, 358

Every one has a right cell reading literally `New Section` in the NCRB table.
Worth confirming a couple — 111 organised crime, 113 terrorist act and 152 are
genuinely new offences with no IPC equivalent.

---

## C. Title diffs, body vs the Act's own index

4 diffs, all expected to be cosmetic. Body text wins. If any is NOT cosmetic,
the parser is wrong and ingest must not be trusted.

| Section | Index says | Body says | Why |
|---|---|---|---|
| 44 | ...innocent person. *(+ stray fragment)* | ...innocent person | index artifact on PDF p4 |
| 179 | ...bank-notes | ...bank-notes | line-break hyphenation |
| 180 | ...bank-notes | ...bank-notes | line-break hyphenation |
| 330 | House-trespass and hous-**eb**reaking | House-trespass and house-breaking | **the PDF's index has a typo** |
