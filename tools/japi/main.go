package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	restish "github.com/rest-sh/restish/v2"
)

const (
	apiBaseURL = "https://jupyter-assistant.dzackgarza.com"
	apiSpecURL = apiBaseURL + "/openapi.json"
)

type envelopeFormatter struct{}

type assistantAPIError struct {
	status  any
	message string
}

func (e *assistantAPIError) Error() string {
	if e.status == nil {
		return e.message
	}
	return fmt.Sprintf("assistant API error (embedded HTTP status %v): %s", e.status, e.message)
}

func (envelopeFormatter) Format(w io.Writer, response *restish.Response, _ bool) error {
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(response.Body); err != nil {
		return err
	}

	body, ok := response.Body.(map[string]any)
	if !ok {
		return nil
	}
	succeeded, hasOK := body["ok"].(bool)
	if !hasOK || succeeded {
		return nil
	}

	message, _ := body["error_message"].(string)
	if message == "" {
		message = "request failed without an error_message"
	}
	return &assistantAPIError{
		status:  body["http_status"],
		message: message,
	}
}

func validateArgs(args []string) error {
	for index, arg := range args {
		switch {
		case arg == "--rsh-config" || strings.HasPrefix(arg, "--rsh-config="):
			return errors.New("japi manages its own temporary Restish config")
		case arg == "-o" || arg == "--rsh-output-format":
			if index+1 >= len(args) {
				return errors.New("output format flag requires a value")
			}
			if args[index+1] != "json" {
				return errors.New("japi fixes output to JSON so ok:false can produce a nonzero exit")
			}
		case strings.HasPrefix(arg, "--rsh-output-format="):
			if strings.TrimPrefix(arg, "--rsh-output-format=") != "json" {
				return errors.New("japi fixes output to JSON so ok:false can produce a nonzero exit")
			}
		}
	}
	return nil
}

func run(args []string) error {
	if err := validateArgs(args); err != nil {
		return err
	}

	stateDirectory, err := os.MkdirTemp("", "japi-")
	if err != nil {
		return fmt.Errorf("create isolated Restish state: %w", err)
	}
	defer os.RemoveAll(stateDirectory)

	configPath := filepath.Join(stateDirectory, "restish.json")
	if err := os.WriteFile(configPath, []byte("{}\n"), 0o600); err != nil {
		return fmt.Errorf("create isolated Restish config: %w", err)
	}
	if err := os.Setenv("RSH_CONFIG", configPath); err != nil {
		return fmt.Errorf("set isolated Restish config: %w", err)
	}
	if err := os.Setenv("RSH_CACHE_DIR", filepath.Join(stateDirectory, "cache")); err != nil {
		return fmt.Errorf("set isolated Restish cache: %w", err)
	}
	if err := os.Setenv("RSH_OUTPUT_FORMAT", "json"); err != nil {
		return fmt.Errorf("set JSON output format: %w", err)
	}

	cli := restish.New()
	cli.SetCommandName("japi")
	cli.SetCommandDescription(
		"Jupyter Assistant API",
		"Live OpenAPI-discovered CLI for the Jupyter Assistant API.",
	)
	cli.SetDefaultConfig(&restish.Config{APIs: map[string]*restish.APIConfig{
		"api": {
			BaseURL: apiBaseURL,
			SpecURL: apiSpecURL,
		},
	}})
	cli.SetCommandSurface(restish.CommandSurface{
		PromotedAPI:         "api",
		HideSupportCommands: true,
	})
	cli.AddFormatter("json", envelopeFormatter{})

	return cli.Run(args)
}

func main() {
	if err := run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
