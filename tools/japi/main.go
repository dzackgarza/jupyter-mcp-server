package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	restish "github.com/rest-sh/restish/v2"
)

const (
	defaultAPIBaseURL = "https://jupyter-assistant.dzackgarza.com"
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

func normalizeBaseURL(raw string) (string, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return "", errors.New("API base URL cannot be empty")
	}
	if !strings.Contains(value, "://") {
		value = "https://" + value
	}
	parsed, err := url.Parse(value)
	if err != nil {
		return "", fmt.Errorf("parse API base URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("API base URL must use http or https, got %q", parsed.Scheme)
	}
	if parsed.Host == "" {
		return "", errors.New("API base URL must include a hostname")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	return parsed.String(), nil
}

func parseLauncherArgs(args []string) (string, []string, error) {
	rawBaseURL := os.Getenv("JAPI_BASE_URL")
	if rawBaseURL == "" {
		rawBaseURL = defaultAPIBaseURL
	}

	forwarded := make([]string, 0, len(args))
	if len(args) > 0 {
		forwarded = append(forwarded, args[0])
	}

	for index := 1; index < len(args); index++ {
		arg := args[index]
		switch {
		case arg == "--":
			forwarded = append(forwarded, args[index:]...)
			index = len(args)
		case arg == "--base-url" || arg == "--hostname":
			if index+1 >= len(args) {
				return "", nil, fmt.Errorf("%s requires a value", arg)
			}
			rawBaseURL = args[index+1]
			index++
		case strings.HasPrefix(arg, "--base-url="):
			rawBaseURL = strings.TrimPrefix(arg, "--base-url=")
		case strings.HasPrefix(arg, "--hostname="):
			rawBaseURL = strings.TrimPrefix(arg, "--hostname=")
		default:
			forwarded = append(forwarded, arg)
		}
	}

	baseURL, err := normalizeBaseURL(rawBaseURL)
	if err != nil {
		return "", nil, err
	}
	return baseURL, forwarded, nil
}

func run(args []string) error {
	apiBaseURL, args, err := parseLauncherArgs(args)
	if err != nil {
		return err
	}
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
			SpecURL: apiBaseURL + "/openapi.json",
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
