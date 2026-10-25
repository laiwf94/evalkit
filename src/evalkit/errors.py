"""Exceptions with stable codes, shared by the CLI and the API."""

from __future__ import annotations


class EvalkitError(Exception):
    code = "internal"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidInputError(EvalkitError):
    code = "invalid_argument"


class NotFoundError(EvalkitError):
    code = "not_found"


class ConflictError(EvalkitError):
    code = "conflict"


class PreconditionError(EvalkitError):
    code = "failed_precondition"
