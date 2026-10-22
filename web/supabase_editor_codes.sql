-- Editor codes: editors mark objects with their email + a code from the admin, without a Supabase account and
-- without any sign-in email. Run once in Supabase -> SQL Editor -> New query -> paste -> Run (safe to run again).
--
--   admin (signed in) presses Add / "New code" in the Editors window -> public.new_editor_code(email) stores a
--   bcrypt hash of a fresh code like 7F3A-91C2-E0B4 and returns the code once; the admin sends it to the editor
--   editor types email + code on the page -> public.editor_code_ok() checks it; every save goes through
--   public.editor_save(), which checks the code again and writes as that editor (decision_log keeps who / when)
-- A new code replaces the old one; "Remove" deletes the editor and with it the code.

create extension if not exists pgcrypto with schema extensions;

alter table public.editors add column if not exists code_hash   text;
alter table public.editors add column if not exists code_set_at timestamptz;

-- who wrote: the signed-in account, or the editor checked by editor_save() (transaction-local setting)
create or replace function public.stamp_decision() returns trigger
  language plpgsql security definer set search_path = public as $$
begin
  new.updated_by := coalesce(nullif(current_setting('review.editor', true), ''), public.current_email());
  new.updated_at := now();
  insert into public.decision_log (object_id, status, note, changed_by)
  values (new.object_id, new.status, new.note, new.updated_by);
  return new;
end $$;

create or replace function public.new_editor_code(p_email text) returns text
  language plpgsql security definer set search_path = public, extensions as $$
declare
  e text := lower(trim(p_email));
  c text;
begin
  if not public.is_admin() then
    raise exception 'only the admin can make editor codes';
  end if;
  if e !~ '^[^@\s]+@[^@\s]+\.[^@\s]+$' then
    raise exception 'not an email address: %', p_email;
  end if;
  c := upper(regexp_replace(encode(gen_random_bytes(6), 'hex'), '^(....)(....)(....)$', '\1-\2-\3'));
  insert into public.editors (email, code_hash, code_set_at) values (e, crypt(c, gen_salt('bf', 8)), now())
    on conflict (email) do update set code_hash = excluded.code_hash, code_set_at = excluded.code_set_at;
  return c;
end $$;

create or replace function public.editor_code_ok(p_email text, p_code text) returns boolean
  language sql stable security definer set search_path = public, extensions as $$
  select exists (select 1 from public.editors e
                 where e.email = lower(trim(p_email)) and e.code_hash is not null
                   and e.code_hash = crypt(upper(trim(p_code)), e.code_hash)) $$;

create or replace function public.editor_save(p_email text, p_code text, p_rows jsonb) returns integer
  language plpgsql security definer set search_path = public, extensions as $$
declare
  n integer;
begin
  if not public.editor_code_ok(p_email, p_code) then
    raise exception 'wrong email or editor code';
  end if;
  perform set_config('review.editor', lower(trim(p_email)), true);
  insert into public.decisions (object_id, status, note)
    select distinct on (r->>'object_id') r->>'object_id', coalesce(r->>'status', ''), left(coalesce(r->>'note', ''), 500)
    from jsonb_array_elements(p_rows) r
    where coalesce(r->>'object_id', '') <> ''
  on conflict (object_id) do update set status = excluded.status, note = excluded.note;
  get diagnostics n = row_count;
  return n;
end $$;

revoke all on function public.new_editor_code(text) from public, anon;
grant execute on function public.new_editor_code(text) to authenticated;
grant execute on function public.editor_code_ok(text, text) to anon, authenticated;
grant execute on function public.editor_save(text, text, jsonb) to anon, authenticated;
