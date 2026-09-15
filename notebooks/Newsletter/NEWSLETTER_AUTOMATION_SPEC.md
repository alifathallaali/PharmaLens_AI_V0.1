# PharmaLens Newsletter Automation Specification

## Goal
Reduce manual work without publishing unchecked AI-generated claims.

## Pipeline
1. Ingest sources
2. Deduplicate
3. Extract entities
4. Classify:
   - company
   - molecule
   - disease
   - regulatory
   - M&A
   - finance
   - market access
   - commercial
   - AI
   - MENA
5. Score:
   - relevance
   - impact
   - evidence quality
   - novelty
   - commercial relevance
6. Generate draft
7. Human editorial approval
8. Publish to Substack
9. Repurpose to LinkedIn
10. Log analytics

## Minimum database objects
source
source_document
company
product
molecule
therapy_area
event
signal
newsletter_issue
content_asset
audience_segment
campaign
analytics_event

## Evidence requirements
Every quantitative claim must have:
- source URL
- source organization
- publication date
- accessed date
- claim text
- confidence/evidence level

## AI must never:
- invent a source
- invent a clinical result
- infer a market share without data
- present an opinion as a fact
- expose private/proprietary PharmaLens customer data
