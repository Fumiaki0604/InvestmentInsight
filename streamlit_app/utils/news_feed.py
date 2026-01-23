from __future__ import annotations

import datetime
import re
from typing import Dict, List
from xml.etree import ElementTree as ET

import requests

ATOM_NS = "{http://www.w3.org/2005/Atom}"
TAG_RE = re.compile(r"<[^>]+>")


def _format_updated(updated: str) -> str:
    if not updated:
        return ""
    try:
        normalized = updated.replace("Z", "+00:00")
        parsed = datetime.datetime.fromisoformat(normalized)
        return parsed.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return updated


def _strip_html(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def parse_atom_feed(xml_data: bytes | str, limit: int = 20) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_data)
    items: List[Dict[str, str]] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = entry.findtext(f"{ATOM_NS}title", default="").strip()
        updated = entry.findtext(f"{ATOM_NS}updated", default="").strip()
        summary = entry.findtext(f"{ATOM_NS}summary", default="").strip()
        link = ""
        for link_elem in entry.findall(f"{ATOM_NS}link"):
            href = link_elem.attrib.get("href")
            rel = link_elem.attrib.get("rel", "alternate")
            if href and rel == "alternate":
                link = href
                break
            if href and not link:
                link = href
        items.append(
            {
                "title": title,
                "link": link,
                "updated": _format_updated(updated),
                "summary": _strip_html(summary),
            }
        )
        if limit and len(items) >= limit:
            break
    return items


def parse_rss_feed(xml_data: bytes | str, limit: int = 20) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_data)
    channel = root.find("channel")
    if channel is None:
        return []
    items: List[Dict[str, str]] = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        items.append(
            {
                "title": title,
                "link": link,
                "updated": pub_date,
                "summary": _strip_html(description),
            }
        )
        if limit and len(items) >= limit:
            break
    return items


def fetch_atom_feed(url: str, timeout_sec: int = 10) -> bytes:
    response = requests.get(url, timeout=timeout_sec)
    response.raise_for_status()
    return response.content


def load_atom_entries(url: str, limit: int = 20) -> List[Dict[str, str]]:
    xml_bytes = fetch_atom_feed(url)
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        raise ValueError(f"XML parse error: {e}") from e
    if root.tag == "rss" or root.tag.endswith("}rss"):
        return parse_rss_feed(xml_bytes, limit)
    return parse_atom_feed(xml_bytes, limit)
