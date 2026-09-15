create table if not exists public.med_rep_calendar_connections (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null,
    provider text not null default 'google',
    calendar_id text default 'primary',
    connected boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.med_rep_call_plans (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null,
    plan_date date not null,
    status text not null default 'draft',
    plan_json jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.med_rep_visits (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null,
    plan_id uuid references public.med_rep_call_plans(id) on delete set null,
    hcp_id text not null,
    hcp_name text,
    hco_name text,
    start_at timestamptz not null,
    end_at timestamptz not null,
    objective text,
    product text,
    location text,
    status text not null default 'planned',
    google_event_id text,
    created_at timestamptz not null default now()
);

create index if not exists idx_med_rep_plans_user_date
on public.med_rep_call_plans(user_id, plan_date);

create index if not exists idx_med_rep_visits_user_start
on public.med_rep_visits(user_id, start_at);

create index if not exists idx_med_rep_visits_hcp
on public.med_rep_visits(hcp_id);

-- IMPORTANT:
-- Do not store Google OAuth access/refresh tokens in plaintext here.
-- Use your existing secure auth/secret storage and RLS policies.
