'use client';

import { forwardRef, InputHTMLAttributes, SelectHTMLAttributes, TextareaHTMLAttributes } from 'react';
import { ExclamationCircleIcon } from '@heroicons/react/24/outline';

// Base props for all form fields
interface BaseFormFieldProps {
  label: string;
  id: string;
  error?: string;
  helperText?: string;
  required?: boolean;
  className?: string;
  labelClassName?: string;
  wrapperClassName?: string;
}

// Input field props
interface InputFieldProps extends BaseFormFieldProps, Omit<InputHTMLAttributes<HTMLInputElement>, 'id' | 'className'> {
  type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url' | 'search';
}

// Select field props
interface SelectFieldProps extends BaseFormFieldProps, Omit<SelectHTMLAttributes<HTMLSelectElement>, 'id' | 'className'> {
  type: 'select';
  options: Array<{ value: string; label: string; disabled?: boolean }>;
  placeholder?: string;
}

// Textarea field props
interface TextareaFieldProps extends BaseFormFieldProps, Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'id' | 'className'> {
  type: 'textarea';
  rows?: number;
}

// Checkbox field props
interface CheckboxFieldProps extends BaseFormFieldProps, Omit<InputHTMLAttributes<HTMLInputElement>, 'id' | 'className' | 'type'> {
  type: 'checkbox';
  description?: string;
}

// Radio field props
interface RadioFieldProps extends BaseFormFieldProps, Omit<InputHTMLAttributes<HTMLInputElement>, 'id' | 'className' | 'type'> {
  type: 'radio';
  options: Array<{ value: string; label: string; description?: string; disabled?: boolean }>;
}

type FormFieldProps = InputFieldProps | SelectFieldProps | TextareaFieldProps | CheckboxFieldProps | RadioFieldProps;

// Input component
const InputField = forwardRef<HTMLInputElement, InputFieldProps>(
  ({ label, id, error, helperText, required, wrapperClassName, labelClassName, ...props }, ref) => {
    const baseInputClasses = `
      block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 
      shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 
      focus:ring-2 focus:ring-inset focus:ring-zinc-600 
      sm:text-sm sm:leading-6 transition-colors
      ${error ? 'ring-red-300 focus:ring-red-600' : ''}
      ${props.disabled ? 'bg-gray-50 text-gray-500 cursor-not-allowed' : ''}
    `;

    return (
      <div className={wrapperClassName}>
        <label htmlFor={id} className={`block text-sm font-medium leading-6 text-gray-900 ${labelClassName || ''}`}>
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        <div className="mt-2 relative">
          <input
            ref={ref}
            id={id}
            aria-describedby={error ? `${id}-error` : helperText ? `${id}-description` : undefined}
            aria-invalid={error ? 'true' : 'false'}
            className={baseInputClasses}
            {...props}
          />
          {error && (
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
              <ExclamationCircleIcon className="h-5 w-5 text-red-500" aria-hidden="true" />
            </div>
          )}
        </div>
        {error && (
          <p className="mt-2 text-sm text-red-600" id={`${id}-error`}>
            {error}
          </p>
        )}
        {helperText && !error && (
          <p className="mt-2 text-sm text-gray-500" id={`${id}-description`}>
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

// Select component
const SelectField = forwardRef<HTMLSelectElement, SelectFieldProps>(
  ({ label, id, error, helperText, required, options, placeholder, wrapperClassName, labelClassName, ...props }, ref) => {
    const baseSelectClasses = `
      block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 
      shadow-sm ring-1 ring-inset ring-gray-300 
      focus:ring-2 focus:ring-inset focus:ring-zinc-600 
      sm:text-sm sm:leading-6 transition-colors
      ${error ? 'ring-red-300 focus:ring-red-600' : ''}
      ${props.disabled ? 'bg-gray-50 text-gray-500 cursor-not-allowed' : ''}
    `;

    return (
      <div className={wrapperClassName}>
        <label htmlFor={id} className={`block text-sm font-medium leading-6 text-gray-900 ${labelClassName || ''}`}>
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        <div className="mt-2 relative">
          <select
            ref={ref}
            id={id}
            aria-describedby={error ? `${id}-error` : helperText ? `${id}-description` : undefined}
            aria-invalid={error ? 'true' : 'false'}
            className={baseSelectClasses}
            {...props}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {options.map((option) => (
              <option key={option.value} value={option.value} disabled={option.disabled}>
                {option.label}
              </option>
            ))}
          </select>
          {error && (
            <div className="absolute inset-y-0 right-8 pr-3 flex items-center pointer-events-none">
              <ExclamationCircleIcon className="h-5 w-5 text-red-500" aria-hidden="true" />
            </div>
          )}
        </div>
        {error && (
          <p className="mt-2 text-sm text-red-600" id={`${id}-error`}>
            {error}
          </p>
        )}
        {helperText && !error && (
          <p className="mt-2 text-sm text-gray-500" id={`${id}-description`}>
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

// Textarea component
const TextareaField = forwardRef<HTMLTextAreaElement, TextareaFieldProps>(
  ({ label, id, error, helperText, required, rows = 4, wrapperClassName, labelClassName, ...props }, ref) => {
    const baseTextareaClasses = `
      block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 
      shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 
      focus:ring-2 focus:ring-inset focus:ring-zinc-600 
      sm:text-sm sm:leading-6 transition-colors resize-vertical
      ${error ? 'ring-red-300 focus:ring-red-600' : ''}
      ${props.disabled ? 'bg-gray-50 text-gray-500 cursor-not-allowed' : ''}
    `;

    return (
      <div className={wrapperClassName}>
        <label htmlFor={id} className={`block text-sm font-medium leading-6 text-gray-900 ${labelClassName || ''}`}>
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        <div className="mt-2 relative">
          <textarea
            ref={ref}
            id={id}
            rows={rows}
            aria-describedby={error ? `${id}-error` : helperText ? `${id}-description` : undefined}
            aria-invalid={error ? 'true' : 'false'}
            className={baseTextareaClasses}
            {...props}
          />
          {error && (
            <div className="absolute top-2 right-2 pointer-events-none">
              <ExclamationCircleIcon className="h-5 w-5 text-red-500" aria-hidden="true" />
            </div>
          )}
        </div>
        {error && (
          <p className="mt-2 text-sm text-red-600" id={`${id}-error`}>
            {error}
          </p>
        )}
        {helperText && !error && (
          <p className="mt-2 text-sm text-gray-500" id={`${id}-description`}>
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

// Checkbox component
const CheckboxField = forwardRef<HTMLInputElement, CheckboxFieldProps>(
  ({ label, id, error, helperText, required, description, wrapperClassName, labelClassName, type, ...props }, ref) => {
    return (
      <div className={wrapperClassName}>
        <div className="relative flex items-start">
          <div className="flex h-6 items-center">
            <input
              ref={ref}
              id={id}
              type="checkbox"
              aria-describedby={error ? `${id}-error` : helperText ? `${id}-description` : undefined}
              aria-invalid={error ? 'true' : 'false'}
              className={`h-4 w-4 rounded border-gray-300 text-zinc-600 focus:ring-zinc-600 transition-colors ${
                error ? 'border-red-300' : ''
              } ${props.disabled ? 'cursor-not-allowed' : ''}`}
              {...props}
            />
          </div>
          <div className="ml-3 text-sm leading-6">
            <label htmlFor={id} className={`font-medium text-gray-900 ${labelClassName || ''}`}>
              {label}
              {required && <span className="text-red-500 ml-1">*</span>}
            </label>
            {description && (
              <p className="text-gray-500">{description}</p>
            )}
          </div>
        </div>
        {error && (
          <p className="mt-2 text-sm text-red-600" id={`${id}-error`}>
            {error}
          </p>
        )}
        {helperText && !error && (
          <p className="mt-2 text-sm text-gray-500" id={`${id}-description`}>
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

// Radio component
const RadioField = forwardRef<HTMLInputElement, RadioFieldProps>(
  ({ label, id, error, helperText, required, options, wrapperClassName, labelClassName, type, ...props }, ref) => {
    return (
      <div className={wrapperClassName}>
        <fieldset>
          <legend className={`text-sm font-medium leading-6 text-gray-900 ${labelClassName || ''}`}>
            {label}
            {required && <span className="text-red-500 ml-1">*</span>}
          </legend>
          <div className="mt-3 space-y-3">
            {options.map((option, index) => (
              <div key={option.value} className="relative flex items-start">
                <div className="flex h-6 items-center">
                  <input
                    ref={index === 0 ? ref : undefined}
                    id={`${id}-${option.value}`}
                    name={props.name || id}
                    type="radio"
                    value={option.value}
                    disabled={option.disabled || props.disabled}
                    aria-describedby={error ? `${id}-error` : helperText ? `${id}-description` : undefined}
                    aria-invalid={error ? 'true' : 'false'}
                    className={`h-4 w-4 border-gray-300 text-zinc-600 focus:ring-zinc-600 transition-colors ${
                      error ? 'border-red-300' : ''
                    } ${option.disabled || props.disabled ? 'cursor-not-allowed' : ''}`}
                    {...props}
                  />
                </div>
                <div className="ml-3 text-sm leading-6">
                  <label htmlFor={`${id}-${option.value}`} className="font-medium text-gray-900">
                    {option.label}
                  </label>
                  {option.description && (
                    <p className="text-gray-500">{option.description}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </fieldset>
        {error && (
          <p className="mt-2 text-sm text-red-600" id={`${id}-error`}>
            {error}
          </p>
        )}
        {helperText && !error && (
          <p className="mt-2 text-sm text-gray-500" id={`${id}-description`}>
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

// Main FormField component
export const FormField = forwardRef<
  HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement,
  FormFieldProps
>((props, ref) => {
  switch (props.type) {
    case 'select':
      return <SelectField ref={ref as any} {...props} />;
    case 'textarea':
      return <TextareaField ref={ref as any} {...props} />;
    case 'checkbox':
      return <CheckboxField ref={ref as any} {...props} />;
    case 'radio':
      return <RadioField ref={ref as any} {...props} />;
    default:
      return <InputField ref={ref as any} {...(props as InputFieldProps)} />;
  }
});

FormField.displayName = 'FormField';
InputField.displayName = 'InputField';
SelectField.displayName = 'SelectField';
TextareaField.displayName = 'TextareaField';
CheckboxField.displayName = 'CheckboxField';
RadioField.displayName = 'RadioField'; 