import React, { useState, useRef, useEffect } from 'react';
import { Check, ChevronDown, Search } from 'lucide-react';

interface Option {
  label: string;
  value: number;
}

interface MultiSelectProps {
  options: Option[];
  value: number[];
  onChange: (value: number[]) => void;
  placeholder?: string;
  label?: string;
  className?: string;
  icon?: React.ElementType;
}

export default function MultiSelect({ 
  options, 
  value, 
  onChange, 
  placeholder = 'Select...', 
  label,
  className = 'w-48',
  icon: Icon
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ top: 0, left: 0, width: 0 });

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      // Check if click is outside container AND outside the fixed dropdown
      const dropdown = document.getElementById(`multiselect-dropdown-${label}`);
      if (
        containerRef.current && 
        !containerRef.current.contains(event.target as Node) &&
        (!dropdown || !dropdown.contains(event.target as Node))
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [label]);

  const updatePosition = () => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setPosition({
        top: rect.bottom + 4,
        left: rect.left,
        width: rect.width
      });
    }
  };

  useEffect(() => {
    if (isOpen) {
      updatePosition();
      window.addEventListener('resize', updatePosition);
      window.addEventListener('scroll', updatePosition, true);
    }
    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
    };
  }, [isOpen]);

  const filteredOptions = options.filter(option => 
    option.label.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const toggleOption = (optionValue: number) => {
    const newValue = value.includes(optionValue)
      ? value.filter(v => v !== optionValue)
      : [...value, optionValue];
    onChange(newValue);
  };

  return (
    <div className="relative h-full" ref={containerRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex flex-col items-center gap-1 p-2 rounded hover:bg-white/10 transition-all ${className} ${
          value.length > 0 ? 'bg-blue-500/20 text-blue-400' : 'text-neutral-400'
        }`}
        title={placeholder}
      >
        <div className="relative">
          {Icon ? <Icon className="w-6 h-6" /> : <ChevronDown className="w-6 h-6" />}
          {value.length > 0 && (
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full border-2 border-[#1a1d21]" />
          )}
        </div>
        <span className="text-[10px] font-medium">{label || placeholder}</span>
      </button>

      {isOpen && (
        <div 
          id={`multiselect-dropdown-${label}`}
          style={{ 
            position: 'fixed', 
            top: position.top, 
            left: position.left, 
            width: Math.max(position.width, 256), // Min width 256px
            zIndex: 9999
          }}
          className="bg-[#1a1d21] border border-white/10 rounded shadow-2xl flex flex-col max-h-80 animate-in fade-in zoom-in-95 duration-100"
        >
          <div className="p-2 border-b border-white/5">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-neutral-500" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search..."
                className="w-full bg-[#0d0e10] border border-white/10 rounded pl-8 pr-2 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/20 placeholder:text-neutral-600"
                autoFocus
              />
            </div>
          </div>
          
          <div className="overflow-y-auto custom-scrollbar flex-1 py-1">
            {filteredOptions.length === 0 ? (
              <div className="px-3 py-2 text-xs text-neutral-500 text-center">No results found</div>
            ) : (
              filteredOptions.map(option => {
                const isSelected = value.includes(option.value);
                return (
                  <div
                    key={option.value}
                    onClick={() => toggleOption(option.value)}
                    className={`flex items-center gap-2 px-3 py-1.5 text-xs cursor-pointer transition-colors ${
                      isSelected ? 'bg-blue-500/10 text-blue-400' : 'text-neutral-300 hover:bg-white/5'
                    }`}
                  >
                    <div className={`flex items-center justify-center w-3 h-3 rounded border ${
                      isSelected ? 'bg-blue-500 border-blue-500' : 'border-white/20'
                    }`}>
                      {isSelected && <Check className="w-2.5 h-2.5 text-white" />}
                    </div>
                    <span>{option.label}</span>
                  </div>
                );
              })
            )}
          </div>
          
          <div className="p-2 border-t border-white/5 flex justify-between">
            <button
              onClick={() => onChange(filteredOptions.map(o => o.value))}
              className="text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
            >
              Select All
            </button>
            <button
              onClick={() => onChange([])}
              className="text-[10px] text-neutral-500 hover:text-neutral-300 transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
