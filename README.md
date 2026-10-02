# Reading school roof condition from field photos

Can a vision model read, in a field photo of a school, the roof condition its principal reported?

## TL;DR

Colombia has little reliable information on the physical condition of its public schools: of 52,823 official school sites, only 15,244 were covered by the national infrastructure survey (FFIE, 2021–22), and full censuses are expensive. This repository tests whether a vision model can read, from a field photo of a school's façade, the roof condition its principal reported in that survey. It uses 35,966 labelled field photos from the FFIE archive, validates by municipality to rule out leakage, and compares a simple image-statistics baseline against a pretrained vision model. The project is at the feasibility stage: data and splits are done, models are not yet trained.

## Why this matters

Knowing which schools are in poor physical condition decides where reconstruction and maintenance money goes, but Colombia mostly does not know. The 2021–22 FFIE survey reached 15,244 of 52,823 official sites. A new national round ran from April to June 2026, aimed at more than 48,000 sites, but its responses are not public. It also predates the August 2026 earthquake, so for the schools that were shaken it describes their condition before the event.

The gap shows up when it matters most. After the M 7.4 earthquake in Chocó on 10 August 2026, 4,350 school sites with 963,176 students were exposed to shaking of intensity VI or higher, and 2,779 of them (63.9%) had never been surveyed. The same blind spot limits planning for climate adaptation and for bringing schools up to current safety and sustainability standards, in Colombia and across Latin America.

Remote data does not close it. In the sibling project `school_infrastructure`, five external sources (municipal poverty, climate, cadastre, historical investment and satellite building footprints) predicted a school's *size* well but not its *condition*: reported roof damage reached an AUC of only 0.582. A school's state cannot be seen from above. Field photos look at the building up close, which makes them the next thing to test. If they work, the same approach could extend to street-level imagery and to other countries in the region.
