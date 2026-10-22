-- Tile (attribute) reviews on the online review page: the review_tile category and the six-attribute review
-- (form, surface finish, transparency, texture, glaze effects, defects) of an object.
-- Run once in Supabase -> SQL Editor -> New query -> paste this file -> Run (safe to run again).
-- Needs supabase_setup.sql and supabase_editor_codes.sql (both already run for this project).
--
-- Who can do what (same rules as the decisions):
--   anyone with the link : reads every tile review
--   editors              : save with their email + editor code (public.tile_save) or as a signed-in editor account
--   every save           : stamped (who / when) and appended to public.tile_review_log
-- The record is the same JSON as review/tile_review.json on the PC (glazeprep/review_tile.py);
-- SYNC_ONLINE_REVIEWS.bat brings it back to the PC and checks every value there.

create table if not exists public.tile_reviews (
  object_id  text primary key check (object_id ~ '^[0-9]+_[0-9]+$'),
  review     jsonb not null default '{}'::jsonb check (jsonb_typeof(review) = 'object' and length(review::text) <= 20000),
  category   text not null default '' check (category in ('', 'review_tile')),
  updated_by text,
  updated_at timestamptz not null default now()
);

create table if not exists public.tile_review_log (
  id         bigserial primary key,
  object_id  text not null,
  review     jsonb not null,
  category   text not null default '',
  changed_by text,
  changed_at timestamptz not null default now()
);

-- who / when on every write (an editor-code save sets review.editor, see tile_save) + history
create or replace function public.stamp_tile_review() returns trigger
  language plpgsql security definer set search_path = public as $$
begin
  new.category := case when new.review ->> 'category' = 'review_tile' then 'review_tile' else '' end;
  new.updated_by := coalesce(nullif(current_setting('review.editor', true), ''), public.current_email());
  new.updated_at := now();
  insert into public.tile_review_log (object_id, review, category, changed_by)
  values (new.object_id, new.review, new.category, new.updated_by);
  return new;
end $$;

drop trigger if exists trg_stamp_tile_review on public.tile_reviews;
create trigger trg_stamp_tile_review before insert or update on public.tile_reviews
  for each row execute function public.stamp_tile_review();

alter table public.tile_reviews    enable row level security;
alter table public.tile_review_log enable row level security;

drop policy if exists "tile: everyone reads"     on public.tile_reviews;
drop policy if exists "tile: editors insert"     on public.tile_reviews;
drop policy if exists "tile: editors update"     on public.tile_reviews;
drop policy if exists "tile log: everyone reads" on public.tile_review_log;

create policy "tile: everyone reads" on public.tile_reviews for select using (true);
create policy "tile: editors insert" on public.tile_reviews for insert to authenticated with check (public.is_editor());
create policy "tile: editors update" on public.tile_reviews for update to authenticated
  using (public.is_editor()) with check (public.is_editor());
create policy "tile log: everyone reads" on public.tile_review_log for select using (true);

-- save with email + editor code (no Supabase account): checks the code, then writes as that editor
create or replace function public.tile_save(p_email text, p_code text, p_rows jsonb) returns integer
  language plpgsql security definer set search_path = public, extensions as $$
declare
  n integer;
begin
  if not public.editor_code_ok(p_email, p_code) then
    raise exception 'wrong email or editor code';
  end if;
  perform set_config('review.editor', lower(trim(p_email)), true);
  insert into public.tile_reviews (object_id, review)
    select distinct on (r ->> 'object_id') r ->> 'object_id', r -> 'review'
    from jsonb_array_elements(p_rows) r
    where coalesce(r ->> 'object_id', '') <> '' and jsonb_typeof(r -> 'review') = 'object'
  on conflict (object_id) do update set review = excluded.review;
  get diagnostics n = row_count;
  return n;
end $$;

grant select on public.tile_reviews, public.tile_review_log to anon, authenticated;
grant insert, update on public.tile_reviews to authenticated;
grant execute on function public.tile_save(text, text, jsonb) to anon, authenticated;
