'''
URL
 ↓
URL Parser + DNS + SSL + HTML Fetch
 ↓
Feature Extraction Functions
 ↓
14 Feature Dict (0 / 1)
 ↓
ML Model
'''
import socket
import ssl
import logging
import requests
import dns.resolver
import whois

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime, timezone
from functools import lru_cache

logger = logging.getLogger(__name__)


def extract_root_domain(domain: str) -> str:
    """
    Extract the registrable root domain from any hostname.

    Examples:
        chat.openai.com  → openai.com
        www.google.com   → google.com
        mail.google.com  → google.com
        github.com       → github.com
        a.b.evil.xyz     → evil.xyz

    This is used so that Abnormal_URL and subdomain checks
    operate on the root domain, not the full subdomain hostname.
    """
    if domain.startswith("www."):
        domain = domain[4:]
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


class FeatureExtractor:
    """
    Extract phishing-related features from a URL in real-time.
    Output values follow dataset convention:
    1 = suspicious / phishing indicator
    0 = legitimate
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (PhishGuard-AI)"
        }

    # -------------------------
    # URL LEVEL FEATURES
    # -------------------------

    def having_ip_address(self, url: str) -> int:
        try:
            socket.inet_aton(urlparse(url).hostname)
            return 1
        except Exception:
            return 0

    def url_length(self, url: str) -> int:
        return 1 if len(url) >= 75 else 0

    def shortening_service(self, url: str) -> int:
        shorteners = ["bit.ly", "tinyurl", "goo.gl", "t.co"]
        return 1 if any(s in url for s in shorteners) else 0

    def having_at_symbol(self, url: str) -> int:
        return 1 if "@" in url else 0

    def double_slash_redirecting(self, url: str) -> int:
        return 1 if url.rfind("//") > 7 else 0

    def prefix_suffix(self, domain: str) -> int:
        return 1 if "-" in domain else 0

    def having_sub_domain(self, domain: str) -> int:
        """
        Smarter subdomain check.

        OLD (buggy): flagged ANY subdomain as suspicious.
            chat.openai.com  → 1  (wrong — one clean subdomain is fine)
            mail.google.com  → 1  (wrong)

        NEW: only flag when there are 2+ subdomains (deep nesting = suspicious).
            chat.openai.com          → 0  (sub.domain.tld = fine)
            login.verify.paypal.com  → 1  (sub.sub.domain.tld = suspicious)
        """
        if domain.startswith("www."):
            domain = domain[4:]
        # After stripping www., legitimate sites have at most 1 dot (domain.tld)
        # or 2 dots for one clean subdomain (sub.domain.tld).
        # 3+ dots means sub.sub.domain.tld → suspicious depth.
        return 1 if domain.count(".") >= 2 else 0

    # -------------------------
    # DNS / SSL / DOMAIN
    # -------------------------

    def dns_record(self, domain: str) -> int:
        try:
            dns.resolver.resolve(domain, "A")
            return 1
        except Exception as e:
            logger.debug(f"dns_record failed for {domain}: {e}")
            return 0

    def ssl_final_state(self, url: str) -> int:
        try:
            hostname = urlparse(url).hostname
            if not hostname:
                return 0
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((hostname, 443))
            ssock = context.wrap_socket(sock, server_hostname=hostname)
            ssock.close()
            return 1
        except Exception as e:
            logger.debug(f"ssl_final_state failed for {url}: {e}")
            return 0

    @lru_cache(maxsize=256)
    def _whois_cached(self, domain: str):
        """Cache WHOIS lookups — domain_registration_length and age_of_domain
        both need WHOIS, this avoids two network calls for the same domain."""
        return whois.whois(domain)

    def domain_registration_length(self, domain: str) -> int:
        try:
            root = extract_root_domain(domain)
            w = self._whois_cached(root)
            cre = w.creation_date
            exp = w.expiration_date
            if isinstance(cre, list):
                cre = cre[0]
            if isinstance(exp, list):
                exp = exp[0]
            if cre and exp:
                if cre.tzinfo and not exp.tzinfo:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp.tzinfo and not cre.tzinfo:
                    cre = cre.replace(tzinfo=timezone.utc)
                days = (exp - cre).days
                return 1 if days >= 365 else 0
        except Exception as e:
            logger.debug(f"domain_registration_length failed for {domain}: {e}")
        return 0

    def age_of_domain(self, domain: str) -> int:
        try:
            root = extract_root_domain(domain)
            w = self._whois_cached(root)
            cd = w.creation_date
            if isinstance(cd, list):
                cd = cd[0]
            if cd:
                now = datetime.now(timezone.utc) if cd.tzinfo else datetime.now()
                days = (now - cd).days
                return 1 if days >= 180 else 0
        except Exception as e:
            logger.debug(f"age_of_domain failed for {domain}: {e}")
        return 0

    # -------------------------
    # HTML BASED FEATURES
    # -------------------------

    def fetch_html(self, url: str) -> str:
        try:
            r = requests.get(
                url,
                timeout=self.timeout,
                headers=self.headers,
                allow_redirects=True,
            )
            return r.text
        except Exception as e:
            logger.debug(f"fetch_html failed for {url}: {e}")
            return ""

    def iframe_present(self, html: str) -> int:
        soup = BeautifulSoup(html, "html.parser")
        return 1 if soup.find("iframe") else 0

    def submitting_to_email(self, html: str) -> int:
        soup = BeautifulSoup(html, "html.parser")
        for f in soup.find_all("form"):
            action = f.get("action", "")
            if "mailto:" in action:
                return 1
        return 0

    def abnormal_url(self, domain: str, html: str) -> int:
        """
        Check whether the site's root domain appears in its own HTML.

        OLD (buggy): checked if full hostname (e.g. 'chat.openai.com') appeared in HTML.
            → Almost always False for subdomain sites, even totally legit ones.
            → chat.openai.com HTML contains 'openai.com', not 'chat.openai.com'.
            → This caused false positives on every subdomain URL.

        FIX: check root domain ('openai.com') instead of full hostname.
            → Legit sites always reference their own root domain in their HTML.
            → Phishing sites (hosted on unrelated domains) won't contain it.
        """
        root = extract_root_domain(domain)
        return 0 if root in html else 1

    def request_url(self, domain: str, html: str) -> int:
        """
        Check if images load from external (non-matching) domains.
        Uses root domain comparison so subdomains of the same site are not flagged.
        e.g. an image on chat.openai.com loaded from cdn.openai.com is fine.
        """
        root = extract_root_domain(domain)
        soup = BeautifulSoup(html, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src and src.startswith("http") and root not in src:
                return 1
        return 0

    # -------------------------
    # MAIN EXTRACTOR
    # -------------------------

    def extract(self, url: str) -> dict:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        html = self.fetch_html(url)

        features = {
            "having_IP_Address":           self.having_ip_address(url),
            "URL_Length":                  self.url_length(url),
            "Shortining_Service":          self.shortening_service(url),
            "having_At_Symbol":            self.having_at_symbol(url),
            "double_slash_redirecting":    self.double_slash_redirecting(url),
            "Prefix_Suffix":               self.prefix_suffix(domain),
            "having_Sub_Domain":           self.having_sub_domain(domain),
            "SSLfinal_State":              self.ssl_final_state(url),
            "Domain_registeration_length": self.domain_registration_length(domain),
            "age_of_domain":               self.age_of_domain(domain),
            "DNSRecord":                   self.dns_record(domain),
            "Submitting_to_email":         self.submitting_to_email(html),
            "Abnormal_URL":                self.abnormal_url(domain, html),
            "Iframe":                      self.iframe_present(html),
        }

        return features