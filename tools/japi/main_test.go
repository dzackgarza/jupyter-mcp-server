package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	restish "github.com/rest-sh/restish/v2"
)

func TestEnvelopeFormatterPreservesSuccess(t *testing.T) {
	var output bytes.Buffer
	err := (envelopeFormatter{}).Format(
		&output,
		&restish.Response{Body: map[string]any{"ok": true, "status": "healthy"}},
		false,
	)

	if err != nil {
		t.Fatalf("format success response: %v", err)
	}
	if !strings.Contains(output.String(), `"ok": true`) {
		t.Fatalf("formatted response omitted body: %s", output.String())
	}
}

func TestEnvelopeFormatterFailsAfterPrintingErrorEnvelope(t *testing.T) {
	var output bytes.Buffer
	err := (envelopeFormatter{}).Format(
		&output,
		&restish.Response{Body: map[string]any{
			"ok":            false,
			"http_status":   float64(404),
			"error_message": "Notebook was not found.",
		}},
		false,
	)

	var apiError *assistantAPIError
	if !errors.As(err, &apiError) {
		t.Fatalf("expected assistantAPIError, got %T: %v", err, err)
	}
	if !strings.Contains(err.Error(), "embedded HTTP status 404") {
		t.Fatalf("error omitted embedded status: %v", err)
	}
	if !strings.Contains(output.String(), `"ok": false`) {
		t.Fatalf("formatted response omitted error envelope: %s", output.String())
	}
}

func TestValidateArgsRejectsNonJSONOutput(t *testing.T) {
	for _, args := range [][]string{
		{"japi", "health", "-o", "yaml"},
		{"japi", "health", "--rsh-output-format=table"},
	} {
		if err := validateArgs(args); err == nil {
			t.Fatalf("expected non-JSON output to be rejected: %v", args)
		}
	}

	if err := validateArgs([]string{"japi", "health", "-o", "json"}); err != nil {
		t.Fatalf("JSON output should be accepted: %v", err)
	}
}

func TestValidateArgsRejectsConfigOverride(t *testing.T) {
	for _, args := range [][]string{
		{"japi", "--rsh-config", "/tmp/restish.json", "health"},
		{"japi", "--rsh-config=/tmp/restish.json", "health"},
	} {
		if err := validateArgs(args); err == nil {
			t.Fatalf("expected config override to be rejected: %v", args)
		}
	}
}
