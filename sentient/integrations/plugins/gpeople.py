"""Google Contacts (People API): search, create, update, delete."""

from __future__ import annotations

from typing import Any

from sentient.integrations.base import IntegrationError, itool
from sentient.integrations.google import gapi
from sentient.integrations.plugins._google_base import GooglePlugin
from sentient.tools.base import Risk, ToolContext

API = "https://people.googleapis.com/v1"
PID = "gpeople"
READ_MASK = "names,emailAddresses,phoneNumbers,organizations,biographies"


def simplify_person(p: dict) -> dict:
    names = p.get("names") or [{}]
    orgs = p.get("organizations") or [{}]
    return {
        "resource_name": p.get("resourceName"),
        "name": names[0].get("displayName"),
        "given_name": names[0].get("givenName"),
        "family_name": names[0].get("familyName"),
        "emails": [e.get("value") for e in p.get("emailAddresses") or [] if e.get("value")],
        "phones": [e.get("value") for e in p.get("phoneNumbers") or [] if e.get("value")],
        "organization": orgs[0].get("name"),
        "title": orgs[0].get("title"),
        "notes": ((p.get("biographies") or [{}])[0]).get("value"),
    }


def _resource(name: str) -> str:
    name = name.strip()
    if not name.startswith("people/"):
        name = f"people/{name}"
    return name


@itool(PID, "gpeople_search_contacts")
async def gpeople_search_contacts(ctx: ToolContext, query: str, max_results: int = 10) -> dict:
    """Find contacts by name, email address or phone number. Returns their resource_name for updates."""
    # Google recommends a warm-up request so the search cache is fresh.
    await gapi(ctx, PID, "GET", f"{API}/people:searchContacts", params={"query": "", "readMask": "names"})
    res = await gapi(ctx, PID, "GET", f"{API}/people:searchContacts", params={
        "query": query, "readMask": READ_MASK, "pageSize": max(1, min(int(max_results or 10), 30))})
    people = [simplify_person(r.get("person") or {}) for r in res.get("results") or []]
    return {"query": query, "count": len(people), "contacts": people}


def _body(given_name: str | None, family_name: str | None, email: str | None, phone: str | None,
          organization: str | None, notes: str | None) -> tuple[dict, list[str]]:
    body: dict[str, Any] = {}
    fields: list[str] = []
    if given_name is not None or family_name is not None:
        body["names"] = [{k: v for k, v in (("givenName", given_name), ("familyName", family_name)) if v is not None}]
        fields.append("names")
    if email is not None:
        body["emailAddresses"] = [{"value": email}] if email else []
        fields.append("emailAddresses")
    if phone is not None:
        body["phoneNumbers"] = [{"value": phone}] if phone else []
        fields.append("phoneNumbers")
    if organization is not None:
        body["organizations"] = [{"name": organization}] if organization else []
        fields.append("organizations")
    if notes is not None:
        body["biographies"] = [{"value": notes, "contentType": "TEXT_PLAIN"}] if notes else []
        fields.append("biographies")
    return body, fields


@itool(PID, "gpeople_create_contact", risk=Risk.write)
async def gpeople_create_contact(ctx: ToolContext, given_name: str, family_name: str | None = None,
                                 email: str | None = None, phone: str | None = None,
                                 organization: str | None = None, notes: str | None = None) -> dict:
    """Add a new contact to Google Contacts."""
    body, _ = _body(given_name, family_name, email, phone, organization, notes)
    res = await gapi(ctx, PID, "POST", f"{API}/people:createContact", params={"personFields": READ_MASK}, json=body)
    return {"created": True, "contact": simplify_person(res)}


@itool(PID, "gpeople_update_contact", risk=Risk.write)
async def gpeople_update_contact(ctx: ToolContext, resource_name: str, given_name: str | None = None,
                                 family_name: str | None = None, email: str | None = None, phone: str | None = None,
                                 organization: str | None = None, notes: str | None = None) -> dict:
    """Change a contact's details (only the fields you pass; email/phone replace the existing ones).
    Get resource_name from gpeople_search_contacts."""
    rn = _resource(resource_name)
    body, fields = _body(given_name, family_name, email, phone, organization, notes)
    if not fields:
        raise IntegrationError("Nothing to change: pass at least one field.")
    current = await gapi(ctx, PID, "GET", f"{API}/{rn}", params={"personFields": READ_MASK})
    if "names" in body:
        existing = (current.get("names") or [{}])[0]
        merged = {"givenName": existing.get("givenName"), "familyName": existing.get("familyName"), **body["names"][0]}
        body["names"] = [{k: v for k, v in merged.items() if v}]
    body["etag"] = current.get("etag")
    res = await gapi(ctx, PID, "PATCH", f"{API}/{rn}:updateContact",
                     params={"updatePersonFields": ",".join(fields), "personFields": READ_MASK}, json=body)
    return {"updated": True, "contact": simplify_person(res)}


@itool(PID, "gpeople_delete_contact", risk=Risk.send)
async def gpeople_delete_contact(ctx: ToolContext, resource_name: str) -> dict:
    """Delete a contact from Google Contacts. Get resource_name from gpeople_search_contacts."""
    rn = _resource(resource_name)
    await gapi(ctx, PID, "DELETE", f"{API}/{rn}:deleteContact")
    return {"deleted": True, "resource_name": rn}


class GPeoplePlugin(GooglePlugin):
    id = PID
    display_name = "Google Contacts"
    description = (
        "Keep your address book organised. Sentient can look up someone's email or phone number, add new "
        "contacts, update details and remove old ones."
    )
    category = "productivity"
    icon = "google-contacts"
    api_name = "People API"
    api_slug = "people.googleapis.com"
    selection_hint = "Use to look up, add, update or delete contacts (emails, phone numbers) in Google Contacts."
    tools = [gpeople_search_contacts, gpeople_create_contact, gpeople_update_contact, gpeople_delete_contact]


PLUGIN = GPeoplePlugin()
