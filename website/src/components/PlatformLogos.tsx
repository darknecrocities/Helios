import React from 'react';

export const OllamaLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="Ollama"
  >
    <path
      d="M11.5 5.5C9.8 5.5 8.5 6.8 8.5 8.5V12.5C7.1 13.8 6.2 15.6 6.2 17.8V26.5C6.2 29.3 8.5 31.5 11.2 31.5H24.8C27.5 31.5 29.8 29.3 29.8 26.5V17.8C29.8 15.6 28.9 13.8 27.5 12.5V8.5C27.5 6.8 26.2 5.5 24.5 5.5C23.1 5.5 22 6.4 21.7 7.7C20.5 7.4 19.3 7.2 18 7.2C16.7 7.2 15.5 7.4 14.3 7.7C14 6.4 12.9 5.5 11.5 5.5Z"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="14" cy="17" r="1.8" fill="currentColor" />
    <circle cx="22" cy="17" r="1.8" fill="currentColor" />
    <path
      d="M15.8 22.2C16.5 23 17.2 23.4 18 23.4C18.8 23.4 19.5 23 20.2 22.2"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
    />
    <path d="M18 19.8V22.2" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
  </svg>
);

export const HuggingFaceLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="Hugging Face"
  >
    <circle cx="18" cy="17.5" r="12" fill="#FFD21E" stroke="#141312" strokeWidth="2" />
    <circle cx="13.8" cy="14.8" r="1.7" fill="#141312" />
    <circle cx="22.2" cy="14.8" r="1.7" fill="#141312" />
    <path
      d="M13.2 19.5C14.3 22.2 16 23.4 18 23.4C20 23.4 21.7 22.2 22.8 19.5"
      stroke="#141312"
      strokeWidth="2.1"
      strokeLinecap="round"
    />
    <path
      d="M5 20.8C3.8 19 4.2 16.8 5.8 16.2C7.2 15.6 8.4 17 9.2 18.5L11.8 21.8C12.6 23 11.9 25.2 10 25.5C8 25.8 6.2 22.6 5 20.8Z"
      fill="#FFD21E"
      stroke="#141312"
      strokeWidth="1.8"
    />
    <path
      d="M31 20.8C32.2 19 31.8 16.8 30.2 16.2C28.8 15.6 27.6 17 26.8 18.5L24.2 21.8C23.4 23 24.1 25.2 26 25.5C28 25.8 29.8 22.6 31 20.8Z"
      fill="#FFD21E"
      stroke="#141312"
      strokeWidth="1.8"
    />
  </svg>
);

export const NvidiaLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="NVIDIA"
  >
    <path
      d="M14.5 9.5C21.2 9.5 26.8 13.8 28.8 18C26.8 22.2 21.2 26.5 14.5 26.5C10.8 26.5 7.4 24.8 5 22.2V28.5H32V7.5H5V13.8C7.4 11.2 10.8 9.5 14.5 9.5Z"
      stroke="#76B900"
      strokeWidth="2.2"
      strokeLinejoin="round"
    />
    <path
      d="M14.5 13.5C18.8 13.5 22.4 15.8 24 18C22.4 20.2 18.8 22.5 14.5 22.5C11.8 22.5 9.4 21 8 18.8C9.4 15.5 11.8 13.5 14.5 13.5Z"
      fill="#76B900"
    />
  </svg>
);

export const AntigravityLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="Google Antigravity"
  >
    <path
      d="M18 4L30 28H24.2L18 15.2L11.8 28H6L18 4Z"
      fill="currentColor"
    />
    <circle cx="18" cy="24.5" r="3.5" fill="#FF5500" />
    <path
      d="M8 19C11.5 15.5 24.5 15.5 28 19"
      stroke="#FF5500"
      strokeWidth="2.2"
      strokeLinecap="round"
    />
  </svg>
);

export const PythonLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="Python"
  >
    <path
      d="M17.8 5C12.5 5 12.8 7.3 12.8 7.3V10.5H18V11.8H10.2C10.2 11.8 6 11.3 6 17.5C6 23.7 9.7 23.5 9.7 23.5H12V20.2C12 20.2 11.8 16.5 15.6 16.5H21.5C21.5 16.5 24.8 16.6 24.8 13.3V7.8C24.8 7.8 25.3 5 17.8 5Z"
      fill="#3776AB"
    />
    <path
      d="M18.2 31C23.5 31 23.2 28.7 23.2 28.7V25.5H18V24.2H25.8C25.8 24.2 30 24.7 30 18.5C30 12.3 26.3 12.5 26.3 12.5H24V15.8C24 15.8 24.2 19.5 20.4 19.5H14.5C14.5 19.5 11.2 19.4 11.2 22.7V28.2C11.2 28.2 10.7 31 18.2 31Z"
      fill="#FFD43B"
    />
    <circle cx="15" cy="8" r="1.2" fill="white" />
    <circle cx="21" cy="28" r="1.2" fill="#141312" />
  </svg>
);

export const PyTorchLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="PyTorch"
  >
    <path
      d="M24.2 11.8C27.6 15.2 27.6 20.8 24.2 24.2C20.8 27.6 15.2 27.6 11.8 24.2C8.4 20.8 8.4 15.2 11.8 11.8L18 5.6V10.5"
      stroke="#EE4C2C"
      strokeWidth="2.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="21.5" cy="13.5" r="2" fill="#EE4C2C" />
  </svg>
);

export const FaissLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="FAISS Vector Search"
  >
    <polygon points="18,5 30,12 30,24 18,31 6,24 6,12" stroke="#0064E0" strokeWidth="2.2" />
    <line x1="18" y1="5" x2="18" y2="31" stroke="#0064E0" strokeWidth="1.8" />
    <line x1="6" y1="12" x2="30" y2="24" stroke="#0064E0" strokeWidth="1.8" />
    <line x1="30" y1="12" x2="6" y2="24" stroke="#0064E0" strokeWidth="1.8" />
    <circle cx="18" cy="18" r="3.5" fill="#FF5500" />
  </svg>
);

export const SqliteLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="SQLite WAL"
  >
    <path
      d="M24 6C17 7 11 14 10 24C9.5 27.5 11 30 13 30C16 30 22 22 26 11C27 8 26 5.8 24 6Z"
      stroke="#003B57"
      strokeWidth="2.3"
      fill="rgba(0,59,87,0.12)"
    />
    <path d="M13 26L25 10" stroke="#FF5500" strokeWidth="2" strokeLinecap="round" />
  </svg>
);

export const OnnxLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="ONNX Runtime"
  >
    <polygon points="18,5 31,15 26,30 10,30 5,15" stroke="currentColor" strokeWidth="2.2" />
    <circle cx="18" cy="5" r="2.5" fill="#FF5500" />
    <circle cx="31" cy="15" r="2.5" fill="currentColor" />
    <circle cx="26" cy="30" r="2.5" fill="currentColor" />
    <circle cx="10" cy="30" r="2.5" fill="currentColor" />
    <circle cx="5" cy="15" r="2.5" fill="#FF5500" />
  </svg>
);

export const PyTorchSafeTensorsLogo: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 32,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 36 36"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-label="SafeTensors"
  >
    <path
      d="M18 5L29 11.2V23.8L18 30L7 23.8V11.2L18 5Z"
      stroke="currentColor"
      strokeWidth="2.2"
    />
    <path d="M18 5V17.5M18 17.5L29 11.2M18 17.5L7 11.2" stroke="#FF5500" strokeWidth="2" />
    <circle cx="18" cy="17.5" r="3" fill="#FF5500" />
  </svg>
);
