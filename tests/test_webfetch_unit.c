/*
 * test_webfetch_unit — web_fetch's deterministic, security-critical parts.
 * No network, no curl. Exhaustively covers the trust-boundary checks (scheme
 * gate, host allowlist incl. the subdomain / look-alike traps), the URL parse,
 * the args extraction, and the HTML stripper. The real curl fetch is in the
 * _int test. No assert() — checks set a flag, exit code carries PASS/FAIL.
 */
#include "test_helpers.h"

#include "../tools/agent_webfetch.h"

#include <stdio.h>
#include <string.h>

static int  fails = 0;
static void check(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static void test_scheme(void) {
    check(webfetch_scheme_ok("http://example.com"), "scheme: http ok");
    check(webfetch_scheme_ok("https://example.com"), "scheme: https ok");
    check(!webfetch_scheme_ok("file:///etc/passwd"), "scheme: file:// rejected");
    check(!webfetch_scheme_ok("ftp://host/x"), "scheme: ftp rejected");
    check(!webfetch_scheme_ok("data:text/html,<b>x</b>"), "scheme: data: rejected");
    check(!webfetch_scheme_ok("gopher://h"), "scheme: gopher rejected");
    check(!webfetch_scheme_ok("javascript:alert(1)"), "scheme: javascript rejected");
    check(!webfetch_scheme_ok("HTTP://example.com"), "scheme: uppercase rejected (strict)");
    check(!webfetch_scheme_ok("//example.com"), "scheme: schemeless rejected");
    check(!webfetch_scheme_ok(""), "scheme: empty rejected");
}

static void test_host(void) {
    char h[256];
    check(webfetch_host("http://example.com/path?q=1", sizeof h, h) &&
                  strcmp(h, "example.com") == 0,
          "host: path stripped");
    check(webfetch_host("https://a.b.c:8443/x", sizeof h, h) && strcmp(h, "a.b.c") == 0,
          "host: port stripped");
    check(webfetch_host("http://h?q", sizeof h, h) && strcmp(h, "h") == 0, "host: query stripped");
    check(webfetch_host("http://h#frag", sizeof h, h) && strcmp(h, "h") == 0,
          "host: fragment stripped");
    check(!webfetch_host("notaurl", sizeof h, h), "host: no scheme -> fail");
}

static void test_allowlist(void) {
    /* nullptr / "" = any host allowed */
    check(webfetch_host_allowed("anything.com", nullptr), "allow: null = any");
    check(webfetch_host_allowed("anything.com", ""), "allow: empty = any");

    const char *al = "example.com, intranet.local";
    check(webfetch_host_allowed("example.com", al), "allow: exact match");
    check(webfetch_host_allowed("www.example.com", al), "allow: subdomain ok");
    check(webfetch_host_allowed("a.b.example.com", al), "allow: deep subdomain ok");
    check(webfetch_host_allowed("intranet.local", al), "allow: second entry, spaces trimmed");
    check(!webfetch_host_allowed("notexample.com", al), "allow: look-alike rejected");
    check(!webfetch_host_allowed("example.com.evil.com", al), "allow: suffix-spoof rejected");
    check(!webfetch_host_allowed("example.org", al), "allow: other TLD rejected");
    check(!webfetch_host_allowed("evil.com", al), "allow: unrelated rejected");
}

static void test_strip(void) {
    char out[1024];
    webfetch_strip_html(
            strlen("<p>Hello <b>world</b></p>"), "<p>Hello <b>world</b></p>", sizeof out, out);
    check(strcmp(out, "Hello world") == 0, "strip: tags removed, text kept");

    const char *attr = "<a href=\"http://x\">link</a>";
    webfetch_strip_html(strlen(attr), attr, sizeof out, out);
    check(strcmp(out, "link") == 0, "strip: attributes (incl. <) dropped");

    const char *ws = "  Lots\n\n  of   \t whitespace  ";
    webfetch_strip_html(strlen(ws), ws, sizeof out, out);
    check(strcmp(out, "Lots of whitespace") == 0, "strip: whitespace collapsed + trimmed");

    webfetch_strip_html(strlen("plain text"), "plain text", sizeof out, out);
    check(strcmp(out, "plain text") == 0, "strip: no-tag passthrough");

    webfetch_strip_html(
            strlen("<div><span></span></div>"), "<div><span></span></div>", sizeof out, out);
    check(out[0] == '\0', "strip: tags-only -> empty");

    /* truncation must stay in-bounds and NUL-terminate */
    char small[8];
    webfetch_strip_html(strlen("abcdefghijklmnop"), "abcdefghijklmnop", sizeof small, small);
    check(strlen(small) < sizeof small, "strip: respects out cap");
}

static void test_args(void) {
    char url[1024];
    check(agent_json_str("{\"url\":\"http://example.com/x\"}", "url", sizeof url, url) &&
                  strcmp(url, "http://example.com/x") == 0,
          "args: url extracted");
    check(!agent_json_str("{\"query\":\"x\"}", "url", sizeof url, url), "args: missing url -> 0");
}

int main(void) {
    test_scheme();
    test_host();
    test_allowlist();
    test_strip();
    test_args();
    if (fails > 0) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return GEIST_TEST_FAIL;
    }
    printf("web_fetch: scheme + allowlist + strip + args pass\n");
    return GEIST_TEST_PASS;
}
