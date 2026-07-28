import { useCallback, useRef, useState } from "react";
import type { DragEvent } from "react";

interface DropzoneProps {
  previewUrl: string | null;
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export function Dropzone({ previewUrl, onFileSelected, disabled }: DropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file && ACCEPTED_TYPES.includes(file.type)) {
        onFileSelected(file);
      }
    },
    [onFileSelected],
  );

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    if (!disabled) handleFiles(event.dataTransfer.files);
  };

  return (
    <div
      className={`dropzone ${isDragging ? "is-dragging" : ""} ${disabled ? "is-disabled" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Subir imagen para analizar"
      onKeyDown={(event) => {
        if ((event.key === "Enter" || event.key === " ") && !disabled) {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="dropzone__input"
        onChange={(event) => handleFiles(event.target.files)}
        disabled={disabled}
      />

      {previewUrl ? (
        <img src={previewUrl} alt="Imagen a analizar" className="dropzone__preview" />
      ) : (
        <div className="dropzone__placeholder">
          <svg viewBox="0 0 24 24" className="dropzone__icon" aria-hidden="true">
            <path
              fill="currentColor"
              d="M12 3l4 4h-3v7h-2V7H8l4-4zM5 19h14v2H5v-2z"
            />
          </svg>
          <p className="dropzone__title">Arrastrá una imagen o hacé clic para elegirla</p>
          <p className="dropzone__hint">JPEG, PNG o WEBP · máx. 10&nbsp;MB</p>
        </div>
      )}
    </div>
  );
}
