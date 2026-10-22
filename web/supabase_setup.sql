-- Ceramic object review: online decisions database (Supabase, free tier).
-- Run once in Supabase -> SQL Editor -> New query -> paste this file -> Run.
-- BEFORE running, replace YOUR_EMAIL@example.com (last lines) with your own email: that account becomes the
-- admin who can add / remove editors from the web page.
--
-- Who can do what:
--   anyone with the link (no sign-in, incognito works) : view photos, crops, colours and every decision
--   signed-in user whose email is in public.editors    : mark Not needed / Needs review, write notes
--   editors with is_admin = true                       : add / remove editor emails (button "Editors" on the page)
-- Every change is also appended to public.decision_log (who, when, what).

create table if not exists public.editors (
  email     text primary key,
  is_admin  boolean not null default false,
  added_at  timestamptz not null default now()
);

create table if not exists public.decisions (
  object_id  text primary key,
  status     text not null default '' check (status in ('', 'not_needed', 'needs_review', 'completed', 'completed_v2')),
  note       text not null default '' check (char_length(note) <= 500),
  updated_by text,
  updated_at timestamptz not null default now()
);

create table if not exists public.decision_log (
  id         bigserial primary key,
  object_id  text not null,
  status     text not null,
  note       text not null default '',
  changed_by text,
  changed_at timestamptz not null default now()
);

-- helpers (security definer, so the policies below can check the editors table without recursion)
create or replace function public.current_email() returns text
  language sql stable as $$ select lower(coalesce(auth.jwt() ->> 'email', '')) $$;

create or replace function public.is_editor() returns boolean
  language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.editors e where lower(e.email) = public.current_email()) $$;

create or replace function public.is_admin() returns boolean
  language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.editors e where lower(e.email) = public.current_email() and e.is_admin) $$;

-- stamp who / when on every write and keep the history
create or replace function public.stamp_decision() returns trigger
  language plpgsql security definer set search_path = public as $$
begin
  new.updated_by := public.current_email();
  new.updated_at := now();
  insert into public.decision_log (object_id, status, note, changed_by)
  values (new.object_id, new.status, new.note, new.updated_by);
  return new;
end $$;

drop trigger if exists trg_stamp_decision on public.decisions;
create trigger trg_stamp_decision before insert or update on public.decisions
  for each row execute function public.stamp_decision();

-- row level security
alter table public.editors      enable row level security;
alter table public.decisions    enable row level security;
alter table public.decision_log enable row level security;

drop policy if exists "decisions: everyone reads"      on public.decisions;
drop policy if exists "decisions: editors insert"      on public.decisions;
drop policy if exists "decisions: editors update"      on public.decisions;
drop policy if exists "log: everyone reads"            on public.decision_log;
drop policy if exists "editors: own row or admin"      on public.editors;
drop policy if exists "editors: admins add"            on public.editors;
drop policy if exists "editors: admins remove"         on public.editors;

create policy "decisions: everyone reads" on public.decisions for select using (true);
create policy "decisions: editors insert" on public.decisions for insert to authenticated with check (public.is_editor());
create policy "decisions: editors update" on public.decisions for update to authenticated
  using (public.is_editor()) with check (public.is_editor());

create policy "log: everyone reads" on public.decision_log for select using (true);

create policy "editors: own row or admin" on public.editors for select to authenticated
  using (lower(email) = public.current_email() or public.is_admin());
create policy "editors: admins add"    on public.editors for insert to authenticated with check (public.is_admin());
create policy "editors: admins remove" on public.editors for delete to authenticated using (public.is_admin());

grant select on public.decisions, public.decision_log to anon, authenticated;
grant insert, update on public.decisions to authenticated;
grant select, insert, delete on public.editors to authenticated;

-- live updates: open pages see other editors' marks without pressing Refresh
do $$ begin
  if not exists (select 1 from pg_publication_tables
                 where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = 'decisions') then
    alter publication supabase_realtime add table public.decisions;
  end if;
end $$;

-- editor codes (sign-in with email + code, no Supabase account): also run web/supabase_editor_codes.sql

-- the first admin: REPLACE with your own email before running
insert into public.editors (email, is_admin) values (lower('YOUR_EMAIL@example.com'), true)
  on conflict (email) do update set is_admin = true;
